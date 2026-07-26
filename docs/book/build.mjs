#!/usr/bin/env node
// Renders docs/book/index.md into a single HTML page inspired by after
// "Dive Into HTML5" (diveintohtml5.info). Zero build dependencies; the body
// font is Linux Libertine, self-hosted from docs/book/fonts/ (OFL/GPL).
//
//   node docs/book/build.mjs            # writes docs/book/index.html
//   node docs/book/build.mjs in.md out.html
//
// Supported markdown subset: # headings (h1–h3), paragraphs, *em* or _em_
// (underscores only at word boundaries), **strong**,
// ++underline++, `code`, ``` fenced code blocks (directly adjacent fences
// with distinct languages render as a page-synced language toggle),
// > blockquotes, -/1. lists
// (indented continuation lines supported), | tables |,
// [links](url), ![images](src), --- horizontal rules (rendered as ε glyphs),
// [TOC] on its own line (replaced with a roman-numeral table of contents
// built from the ## headings), [SIM] / [SIM-FORGET] / [BENCH] on their own
// lines (replaced with the live WASM demos; see sim.js, sim-forget.js,
// bench.js with bench-core.js, and `make book-wasm`), [WIP] on its own
// line (an "under construction" note for an unwritten section), and
// [VERSION] on its own line (the engine version from core/Cargo.toml).

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const inFile = process.argv[2] ?? join(here, "index.md");
const outFile = process.argv[3] ?? join(here, "index.html");

// The engine version, from the core crate — the source of truth.
const version = readFileSync(join(here, "../../core/Cargo.toml"), "utf8")
  .match(/^version\s*=\s*"([^"]+)"/m)[1];

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
   .replace(/"/g, "&quot;");

const slugify = (s) =>
  s.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // links slug by their label
    .toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");

// Inline formatting. Code spans are pulled out first so no other rule
// touches their contents.
function inline(text) {
  const stash = [];
  let out = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => {
    stash.push(`<code>${code}</code>`);
    return `\u0000${stash.length - 1}\u0000`;
  });
  out = out
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1">')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/(^|[\s(>])_([^_\n]+)_(?=$|[\s<).,;:!?…—–-])/g, "$1<em>$2</em>")
    .replace(/\+\+([^+]+)\+\+/g, "<u>$1</u>");
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[i]);
}

// Minimal syntax highlighting for the languages the book uses; anything
// else renders unhighlighted. Tokens: comment, string, keyword, number.
const TOKENS = {
  python: {
    comment: /#[^\n]*/,
    string: /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/,
    keyword: /\b(?:import|from|def|return|for|while|if|elif|else|None|True|False|and|or|not|in|is|class|with|as|lambda|pass|raise|try|except|yield|del|global)\b/,
    number: /\b\d[\d_]*(?:\.\d[\d_]*)?\b/,
  },
  js: {
    comment: /\/\/[^\n]*|\/\*[\s\S]*?\*\//,
    string: /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`/,
    keyword: /\b(?:const|let|var|function|return|await|async|import|from|export|default|new|if|else|for|of|in|while|class|true|false|null|undefined|typeof)\b/,
    number: /\b\d[\d_]*(?:\.\d[\d_]*)?\b/,
  },
};
TOKENS.py = TOKENS.python;
TOKENS.javascript = TOKENS.js;
TOKENS.sh = {
  comment: /#[^\n]*/,
  string: /"(?:\\.|[^"\\\n])*"|'[^'\n]*'/,
  keyword: /\bgittins\b/,
  number: /\b\d[\d_]*(?:\.\d[\d_]*)?\b/,
};
TOKENS.bash = TOKENS.sh;
TOKENS.shell = TOKENS.sh;
// The install box's own tabs. They are separate languages so that box can be
// labelled by install method ("npm", "Browser") without renaming the `js`
// tabs the rest of the book uses for the WASM API.
TOKENS.npm = TOKENS.sh;
TOKENS.browser = TOKENS.js;

function highlight(code, lang) {
  const t = TOKENS[lang];
  if (!t) return escapeHtml(code);
  const re = new RegExp(
    `(${t.comment.source})|(${t.string.source})|(${t.keyword.source})|(${t.number.source})`,
    "g",
  );
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += escapeHtml(code.slice(last, m.index));
    const cls = m[1] ? "com" : m[2] ? "str" : m[3] ? "kw" : "num";
    out += `<span class="tok-${cls}">${escapeHtml(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + escapeHtml(code.slice(last));
}

function render(md) {
  const lines = md.split(/\r?\n/);
  const html = [];
  const chapters = []; // { id, title } from ## headings, for the TOC
  let title = "Untitled";
  let subtitle = "";
  let para = [];
  let list = null; // { tag, items }

  const flushPara = () => {
    if (para.length) {
      html.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      html.push(`<${list.tag}>`);
      for (const item of list.items) html.push(`<li>${inline(item)}</li>`);
      html.push(`</${list.tag}>`);
      list = null;
    }
  };
  const flush = () => { flushPara(); flushList(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Fenced code block. Directly adjacent fences (blank lines only between)
    // with distinct languages are grouped into a tabbed language toggle.
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      flush();
      const blocks = [];
      let lang = fence[1];
      for (;;) {
        const code = [];
        while (++i < lines.length && !/^```\s*$/.test(lines[i])) code.push(lines[i]);
        blocks.push({ lang, code: code.join("\n") });
        let j = i + 1;
        while (j < lines.length && /^\s*$/.test(lines[j])) j++;
        const next = j < lines.length && lines[j].match(/^```(\w*)\s*$/);
        if (!next || !next[1] || blocks.some((b) => b.lang === next[1])) break;
        i = j;
        lang = next[1];
      }
      const codeHtml = (b) =>
        `<code class="language-${b.lang || "text"}">${highlight(b.code, b.lang)}</code>`;
      if (blocks.length > 1) {
        const labels = { python: "Python", py: "Python", js: "WASM (JavaScript)",
          javascript: "WASM (JavaScript)", rust: "Rust", sh: "Shell", bash: "Shell",
          npm: "npm", browser: "Browser" };
        html.push('<div class="codetabs">');
        html.push('<div class="codetabs-nav">' + blocks.map((b, k) =>
          `<button data-lang="${b.lang}"${k === 0 ? ' class="active"' : ""}>` +
          `${labels[b.lang] ?? b.lang}</button>`).join("") + "</div>");
        blocks.forEach((b, k) => {
          html.push(`<pre data-lang="${b.lang}"${k > 0 ? " hidden" : ""}>${codeHtml(b)}</pre>`);
        });
        html.push("</div>");
      } else {
        html.push(`<pre>${codeHtml(blocks[0])}</pre>`);
      }
      continue;
    }

    if (/^\s*$/.test(line)) { flush(); continue; }

    if (/^\[TOC\]\s*$/.test(line)) { flush(); html.push("\u0000TOC\u0000"); continue; }

    // [SIM]: mount point for the regret simulation (docs/book/sim.js).
    if (/^\[SIM\]\s*$/.test(line)) {
      flush();
      html.push('<div id="bandit-sim" class="bandit-sim">' +
        "<noscript>The live simulation requires JavaScript.</noscript></div>");
      html.push('<script type="module" src="sim.js"></script>');
      continue;
    }

    // [VERSION]: the engine version, read from core/Cargo.toml at build time.
    if (/^\[VERSION\]\s*$/.test(line)) {
      flush();
      html.push(`<div class="version">${version}</div>`);
      continue;
    }

    // [WIP]: an under-construction note for an unwritten section.
    if (/^\[WIP\]\s*$/.test(line)) {
      flush();
      html.push('<div class="wip-note" role="note">under construction</div>');
      continue;
    }

    // [BENCH]: the live benchmark table (docs/book/bench.js).
    if (/^\[BENCH\]\s*$/.test(line)) {
      flush();
      html.push('<div id="bandit-bench" class="bandit-sim">' +
        "<noscript>The live benchmark requires JavaScript.</noscript></div>");
      html.push('<script type="module" src="bench.js"></script>');
      continue;
    }

    // [SIM-FORGET]: the non-stationarity demo (docs/book/sim-forget.js).
    if (/^\[SIM-FORGET\]\s*$/.test(line)) {
      flush();
      html.push('<div id="bandit-sim-forget" class="bandit-sim">' +
        "<noscript>The live simulation requires JavaScript.</noscript></div>");
      html.push('<script type="module" src="sim-forget.js"></script>');
      continue;
    }

    if (/^(-{3,}|\*{3,})\s*$/.test(line)) {
      flush();
      html.push('<div class="fleuron" role="separator">ε</div>');
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      flush();
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      if (level === 1) {
        title = text;
        // A lone emphasized paragraph right after the h1 becomes the subtitle.
        let j = i + 1;
        while (j < lines.length && /^\s*$/.test(lines[j])) j++;
        const sub = lines[j]?.match(/^\*([^*].*)\*$/);
        if (sub) { subtitle = sub[1]; i = j; }
        html.push(`<header><h1>${inline(text)}</h1>` +
          (subtitle ? `<p class="subtitle">${inline(subtitle)}</p>` : "") +
          "</header>");
      } else {
        // A "By ..." h2 is the byline, not a chapter: keep it out of the TOC.
        if (level === 2 && !/^by\s/i.test(text)) chapters.push({ id, title: text });
        html.push(`<h${level} id="${id}">${inline(text)}</h${level}>`);
      }
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flush();
      const collected = [quote[1]];
      while (i + 1 < lines.length && /^>\s?/.test(lines[i + 1])) {
        collected.push(lines[++i].replace(/^>\s?/, ""));
      }
      html.push(`<blockquote><p>${inline(collected.join(" "))}</p></blockquote>`);
      continue;
    }

    const ul = line.match(/^[-*]\s+(.*)$/);
    const ol = line.match(/^\d+\.\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul ?? ol)[1]);
      continue;
    }

    // An indented line while a list is open continues the previous item.
    if (list && /^\s+\S/.test(line)) {
      list.items[list.items.length - 1] += " " + line.trim();
      continue;
    }

    if (/^\|.*\|\s*$/.test(line)) {
      flush();
      const rows = [line];
      while (i + 1 < lines.length && /^\|.*\|\s*$/.test(lines[i + 1])) rows.push(lines[++i]);
      const cells = (r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim());
      const hasHeader = rows.length > 1 && /^[\s|:-]+$/.test(rows[1]);
      html.push('<div class="table-wrap"><table>');
      rows.forEach((row, idx) => {
        if (hasHeader && idx === 1) return;
        const tag = hasHeader && idx === 0 ? "th" : "td";
        html.push("<tr>" + cells(row).map((c) => `<${tag}>${inline(c)}</${tag}>`).join("") + "</tr>");
      });
      html.push("</table></div>");
      continue;
    }

    para.push(line.trim());
  }
  flush();

  const toc = chapters.length
    ? '<nav class="toc"><h2>Table of Contents</h2><ol>' +
      chapters.map((c) => `<li><a href="#${c.id}">${inline(c.title)}</a></li>`).join("") +
      "</ol></nav>"
    : "";

  const body = html.join("\n").replace("\u0000TOC\u0000", toc);
  return { title, body };
}

const css = `
  @font-face {
    font-family: "Linux Libertine";
    src: local("Linux Libertine O"), local("Linux Libertine"),
         url("fonts/LinLibertine_Rah.ttf") format("truetype");
    font-weight: normal;
    font-style: normal;
  }
  @font-face {
    font-family: "Linux Libertine";
    src: local("Linux Libertine O Bold"),
         url("fonts/LinLibertine_RBah.ttf") format("truetype");
    font-weight: bold;
    font-style: normal;
  }
  @font-face {
    font-family: "Linux Libertine";
    src: local("Linux Libertine O Italic"),
         url("fonts/LinLibertine_RIah.ttf") format("truetype");
    font-weight: normal;
    font-style: italic;
  }
  @font-face {
    font-family: "Linux Libertine";
    src: local("Linux Libertine O Bold Italic"),
         url("fonts/LinLibertine_RBIah.ttf") format("truetype");
    font-weight: bold;
    font-style: italic;
  }
  html { background: #fff; }
  body {
    font: large/1.567 "Linux Libertine", Georgia, serif;
    color: #1a1a1a;
    max-width: 42em;
    margin: 0 auto;
    padding: 3em 1.5em 6em;
    text-rendering: optimizeLegibility;
  }
  header { text-align: center; margin: 2em 0 0; }
  h1 {
    font-size: 5em;
    font-weight: normal;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1.05;
    margin: 0 0 0.15em;
  }

  /* The byline: the h2 directly after the masthead. */
  header + h2 { margin: 2em 0 4.5em; }
  header + h2:has(+ .version) { margin-bottom: 0.75em; }
  .version {
    text-align: center;
    font-size: 0.8em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #999;
    margin: 0 0 4.5em;
  }
  .subtitle { font-style: italic; color: #555; margin: 0; }
  h2 {
    font-size: 1.7em;
    font-weight: normal;
    text-align: center;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 2.5em 0 1em;
  }
  h3 {
    font-size: 1.2em;
    font-weight: normal;
    font-style: italic;
    margin: 2em 0 0.75em;
  }
  p { margin: 0 0 3em; text-align: justify; hyphens: auto; }

  a {
    color: #900;
    text-decoration: underline dotted;
    text-decoration-thickness: 1px;
    text-underline-offset: 0.15em;
  }
  a:visited { color: #600; }
  a:hover { color: #c00; }
  u {
    text-decoration: underline dotted;
    text-decoration-thickness: 1px;
    text-underline-offset: 0.15em;
  }

  blockquote {
    margin: 1.5em 2em;
    font-style: italic;
    color: #444;
  }
  blockquote p { text-align: left; }

  code {
    font: 0.68em/1.45 "SF Mono", Menlo, Consolas, monospace;
    background: #f4f0e8;
    padding: 0.1em 0.25em;
  }
  pre {
    background: #f4f0e8;
    border: 1px solid #e0d8c8;
    padding: 1em 1.25em;
    overflow-x: auto;
    margin: 0 0 3em;
  }
  pre code { background: none; padding: 0; }

  .tok-com { color: #8a8577; font-style: italic; }
  .tok-str { color: #7a5230; }
  .tok-kw { color: #900; }
  .tok-num { color: #1e5a8a; }

  .codetabs { margin: 0 0 3em; }
  .codetabs pre { margin: 0; }
  .codetabs-nav button {
    font-family: inherit;
    font-size: 0.8em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: none;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 0.3em 0.9em;
    color: #777;
    cursor: pointer;
  }
  .codetabs-nav button.active { color: #900; border-bottom-color: #900; }
  .codetabs-nav button:hover { color: #c00; }

  /* The install box, and only it: the one codetabs group that directly
     follows the version badge. It is a two-word command under the title
     rather than a listing to read, so it is centred and held narrow instead
     of running the full measure like every other block. */
  .version + .codetabs {
    /* Wide enough for the longest tab (the esm.sh URL), which is what sets
       the floor here: a hidden tab contributes nothing to layout, so sizing
       to content would make the box jump width as the reader switches. Still
       well under the full measure the other code blocks run to. */
    max-width: 28em;
    margin-left: auto;
    margin-right: auto;
  }
  .version + .codetabs .codetabs-nav { text-align: center; }
  .version + .codetabs pre { text-align: center; }

  ul, ol { margin: 0 0 3em; padding-left: 2.5em; padding-right: 2em; }
  ol { list-style: upper-roman; }
  ol li::marker { color: #900; }
  li { margin-bottom: 0.25em; }

  .fleuron {
    text-align: center;
    font-size: 1.6em;
    font-style: italic;
    color: #900;
    margin: 2em 0;
    user-select: none;
  }

  .bandit-sim .table-wrap { margin-bottom: 0.75em; }
  .bench-table { margin: 0 auto; font-size: 0.8em; }
  .bench-table td:not(:first-child), .bench-table th:not(:first-child) {
    text-align: right;
  }
  .sim-btn:disabled { color: #aaa; cursor: default; }

  .wip-note {
    text-align: center;
    font-style: italic;
    color: #999;
    font-size: 0.85em;
    margin: 0 0 3em;
  }

  .demo-fallback {
    font-style: italic;
    color: #555;
    border: 1px dashed #ccc;
    padding: 1em 1.25em;
  }

  .bandit-sim { margin: 0 0 3em; text-align: center; }
  .sim-canvas {
    display: block;
    width: 100%;
    height: 240px;
    margin: 0 0 0.5em;
  }
  .sim-line {
    font-size: 0.85em;
    color: #555;
    text-align: center;
    margin: 0;
  }
  .sim-nowrap { white-space: nowrap; }
  .sim-btn {
    font: inherit;
    background: none;
    border: 0;
    padding: 0;
    color: #900;
    text-decoration: underline dotted;
    text-underline-offset: 0.15em;
    cursor: pointer;
  }
  .sim-btn:hover { color: #c00; }
  .sim-read { color: #777; }
  .sim-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    margin-right: 0.35em;
    vertical-align: baseline;
  }
  .sim-swatch-line {
    height: 2px;
    background: #777;
    vertical-align: middle;
  }

  .toc { margin: 3em 0; }
  .toc h2 { margin-top: 0; }
  .toc ol {
    list-style: upper-roman;
    padding: 0;
    margin: 0 auto;
    width: fit-content;
  }
  .toc li { margin: 0.4em 0; padding-left: 0.5em; }
  .toc a { text-decoration: none; }
  .toc a:hover { text-decoration: underline; }

  .table-wrap { overflow-x: auto; margin: 0 0 3em; }
  table { border-collapse: collapse; margin: 0 auto; }
  th, td { border: 1px solid #ccc; padding: 0.4em 0.9em; text-align: left; }
  th {
    font-weight: normal;
    font-variant: small-caps;
    letter-spacing: 0.03em;
    background: #f4f0e8;
  }

  img { max-width: 100%; }
  hr { border: 0; }
`;

const { title, body } = render(readFileSync(inFile, "utf8"));

// Pinned OG/meta description — never derive this from page content.
const description =
  "Gittins is an opinionated, highly optimised contextual bandit engine.";

// Open Graph URLs must be absolute or link scrapers reject them. The
// deployed origin is the default; BOOK_URL overrides it (e.g. previews).
const siteUrl = (process.env.BOOK_URL ?? "https://docs.getgittins.dev")
  .replace(/\/+$/, "");
const meta = [
  `<meta name="description" content="${escapeHtml(description)}">`,
  '<meta property="og:type" content="website">',
  `<meta property="og:title" content="${escapeHtml(title)}">`,
  `<meta property="og:description" content="${escapeHtml(description)}">`,
  `<meta property="og:image" content="${siteUrl}/og.png">`,
  '<meta property="og:image:width" content="1200">',
  '<meta property="og:image:height" content="630">',
  '<meta property="og:image:alt" content="A dark red italic epsilon">',
  ...(siteUrl ? [`<meta property="og:url" content="${siteUrl}/">`] : []),
  '<meta name="twitter:card" content="summary_large_image">',
].join("\n");

writeFileSync(outFile, `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
${meta}
<style>${css}</style>
</head>
<body>
${body}
<div class="fleuron">ε ε ε</div>
<script>
for (const btn of document.querySelectorAll(".codetabs-nav button")) {
  btn.addEventListener("click", () => {
    const lang = btn.dataset.lang;
    for (const group of document.querySelectorAll(".codetabs")) {
      if (!group.querySelector('pre[data-lang="' + lang + '"]')) continue;
      for (const p of group.querySelectorAll("pre[data-lang]")) p.hidden = p.dataset.lang !== lang;
      for (const b of group.querySelectorAll(".codetabs-nav button")) {
        b.classList.toggle("active", b.dataset.lang === lang);
      }
    }
  });
}
</script>
</body>
</html>
`);

console.log(`Wrote ${outFile}`);

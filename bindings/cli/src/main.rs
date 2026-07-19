//! The `gittins` binary: verify / eval / sweep / replay over one
//! experience log (spec/ope.md). Presentation only — flags, tables, exit
//! codes; every semantic lives in `ope.rs`, the reference's port.
//!
//! Exit codes: 0 ok; 1 `verify` found problems; 2 usage or input error.

use gittins_cli::ope::{evaluate, read_log, replay, verify, Event, OpeReport};

const USAGE: &str = "\
usage:
  gittins verify --log FILE[.gz]
  gittins eval   --log FILE --bits N [--epsilon X] [--forgetfulness X]
  gittins sweep  --log FILE --bits N[,N...] [--epsilon X[,X...]] [--forgetfulness X[,X...]]
  gittins replay --log FILE --bits N --horizon SECONDS [--default-reward X]
                 [--epsilon X] [--forgetfulness X]

The log is JSONL (one decision or resolution event per line, spec/ope.md),
gzip-compressed or plain — detected by content, not file name.

  verify  recompute everything the log claims; exit 1 on any violation
  eval    progressive IPS/SNIPS estimate of one configuration, with the
          diagnostics (ess, max weight) an estimate must never be read without
  sweep   eval over a configuration grid, as one markdown table
  replay  rebuild a deployable state from the log; hex on stdout";

const DEFAULT_EPSILON: f64 = gittins_core::exploration::DEFAULT_EPSILON;
const DEFAULT_FORGETTING: f64 = gittins_core::model::DEFAULT_FORGETTING;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match run(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}");
            2
        }
    };
    std::process::exit(code);
}

fn run(args: &[String]) -> Result<i32, String> {
    let Some(command) = args.first() else {
        eprintln!("{USAGE}");
        return Ok(2);
    };
    let flags = Flags::parse(&args[1..])?;
    match command.as_str() {
        "verify" => cmd_verify(&flags),
        "eval" => cmd_eval(&flags),
        "sweep" => cmd_sweep(&flags),
        "replay" => cmd_replay(&flags),
        "--help" | "-h" | "help" => {
            println!("{USAGE}");
            Ok(0)
        }
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

struct Flags(Vec<(String, String)>);

impl Flags {
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut flags = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let name = args[i]
                .strip_prefix("--")
                .ok_or(format!("expected a --flag, got '{}'\n{USAGE}", args[i]))?;
            let value = args
                .get(i + 1)
                .ok_or(format!("--{name} needs a value\n{USAGE}"))?;
            flags.push((name.to_string(), value.clone()));
            i += 2;
        }
        Ok(Flags(flags))
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.get(name).ok_or(format!("--{name} is required\n{USAGE}"))
    }

    fn f64_or(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{name}: '{v}' is not a number")),
        }
    }

    fn bits(&self) -> Result<u32, String> {
        let v = self.required("bits")?;
        v.parse().map_err(|_| format!("--bits: '{v}' is not an integer"))
    }

    fn list<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<Vec<T>, String> {
        match self.get(name) {
            None => Ok(vec![default]),
            Some(v) => v
                .split(',')
                .map(|item| item.trim().parse().map_err(|_| format!("--{name}: '{item}' is not valid")))
                .collect(),
        }
    }

    /// The log as a lazy event stream (ope::read_log): commands walk it
    /// once in constant memory; `sweep` calls this per configuration,
    /// re-reading the file instead of ever holding it.
    fn events(&self) -> Result<Box<dyn Iterator<Item = Result<Event, String>>>, String> {
        read_log(self.required("log")?)
    }
}

fn cmd_verify(flags: &Flags) -> Result<i32, String> {
    let (mut decisions, mut resolutions) = (0usize, 0usize);
    let counted = flags.events()?.map(|event| {
        match &event {
            Ok(Event::Decision(_)) => decisions += 1,
            Ok(Event::Resolution(_)) => resolutions += 1,
            Err(_) => {}
        }
        event
    });
    let problems = verify(counted)?;
    for problem in &problems {
        println!("{problem}");
    }
    if problems.is_empty() {
        println!("ok: {decisions} decisions, {resolutions} resolutions, nothing to report");
        Ok(0)
    } else {
        println!(
            "{} problem(s) in {decisions} decisions, {resolutions} resolutions",
            problems.len()
        );
        Ok(1)
    }
}

fn opt(v: Option<f64>) -> String {
    v.map_or("n/a".to_string(), |x| x.to_string())
}

fn cmd_eval(flags: &Flags) -> Result<i32, String> {
    let report = evaluate(
        flags.events()?,
        flags.bits()?,
        flags.f64_or("epsilon", DEFAULT_EPSILON)?,
        flags.f64_or("forgetfulness", DEFAULT_FORGETTING)?,
    )?;
    println!("decisions    {}", report.decisions);
    println!("resolved     {}", report.resolved);
    println!("censored     {}", report.censored);
    println!("unresolved   {}", report.unresolved);
    println!("logged mean  {}", opt(report.logged_mean));
    println!("ips          {}", opt(report.ips));
    println!("snips        {}", opt(report.snips));
    println!("ess          {}", opt(report.ess));
    println!("max weight   {}", opt(report.max_weight));
    Ok(0)
}

fn cmd_sweep(flags: &Flags) -> Result<i32, String> {
    let all_bits: Vec<u32> = {
        let v = flags.required("bits")?;
        v.split(',')
            .map(|item| item.trim().parse().map_err(|_| format!("--bits: '{item}' is not an integer")))
            .collect::<Result<_, String>>()?
    };
    let epsilons = flags.list("epsilon", DEFAULT_EPSILON)?;
    let forgetfulnesses = flags.list("forgetfulness", DEFAULT_FORGETTING)?;
    println!("| bits | epsilon | forgetfulness | logged mean | ips | snips | ess | max weight | resolved |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &bits in &all_bits {
        for &epsilon in &epsilons {
            for &forgetfulness in &forgetfulnesses {
                let r: OpeReport = evaluate(flags.events()?, bits, epsilon, forgetfulness)?;
                println!(
                    "| {bits} | {epsilon} | {forgetfulness} | {} | {} | {} | {} | {} | {} |",
                    opt(r.logged_mean),
                    opt(r.ips),
                    opt(r.snips),
                    opt(r.ess),
                    opt(r.max_weight),
                    r.resolved,
                );
            }
        }
    }
    Ok(0)
}

fn cmd_replay(flags: &Flags) -> Result<i32, String> {
    let horizon = {
        let v = flags.required("horizon")?;
        v.parse().map_err(|_| format!("--horizon: '{v}' is not a number"))?
    };
    let state = replay(
        flags.events()?,
        flags.bits()?,
        horizon,
        flags.f64_or("default-reward", 0.0)?,
        flags.f64_or("epsilon", DEFAULT_EPSILON)?,
        flags.f64_or("forgetfulness", DEFAULT_FORGETTING)?,
    )?;
    eprintln!(
        "replayed {} observations into a fresh state; hex on stdout",
        state.model_version
    );
    println!("{}", gittins_core::api::serialize(&state));
    Ok(0)
}

export function make_record(bits, context, candidates, decision_id, t, candidate_hash, chosen, flat, propensity, model_version, salt) {
  const features = new Array(flat.length / 2);
  for (let i = 0; i < features.length; i++) features[i] = [flat[2 * i], flat[2 * i + 1]];
  return { bits, context, candidates, decision_id, t, candidate_hash, chosen, features, propensity, model_version, salt };
}
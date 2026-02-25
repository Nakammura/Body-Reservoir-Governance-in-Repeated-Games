// ==========================================================================
// phase_transition.rs — Joint (d, τ_env) phase transition grid
//
// For each combination of reservoir dimension d and defection block length L,
// run the dynamic sentinel and a TfT baseline, measuring:
//   - Detection time (steps from defection start to α dropping below threshold)
//   - Sentinel cumulative payoff
//   - TfT cumulative payoff
//
// This produces the data for the phase diagram: (d, L) plane showing
// "body trust" vs "cognition-dependent" regions.
//
// Schedule: Coop(500) → Defect(L) → Coop(500)
//
// Output CSV:
//   d,defect_length,seed,detection_time,alpha_min_reached,recovery_time,sentinel_payoff,tft_payoff
// ==========================================================================

mod esn;
use esn::*;

fn payoff(a_i: f64, a_j: f64) -> f64 {
    let r = 3.0;
    let s = 0.0;
    let t = 5.0;
    let p = 1.0;
    r * a_i * a_j + s * a_i * (1.0 - a_j) + t * (1.0 - a_i) * a_j + p * (1.0 - a_i) * (1.0 - a_j)
}

fn warmup_cooperative(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) {
    let inp = vec![1.0, 1.0];
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
    }
}

fn run_coupled_dynamic(
    esn: &mut ESN,
    sentinel: &mut Sentinel,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>) {
    // Returns (alpha_trace, actions)
    let n = opp_actions.len();
    let mut alpha_trace = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let a_body = body_readout(&esn.state, w_out, b_out);
        let a_cog = last_opp;
        sentinel.update(&esn.state, a_body, a_cog);
        let alpha = sentinel.alpha;
        let action = alpha * a_body + (1.0 - alpha) * a_cog;
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);
        alpha_trace.push(alpha);
        actions.push(action);
        last_opp = a_opp;
    }
    (alpha_trace, actions)
}

fn run_tft(
    esn: &mut ESN,
    _w_out: &Vec1D,
    _b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    // Pure TfT (α=0): action = a_cog = opponent's last action
    let n = opp_actions.len();
    let mut actions = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let action = last_opp; // TfT
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);
        actions.push(action);
        last_opp = a_opp;
    }
    actions
}

fn init_sentinel_baselines(
    esn: &ESN,
    sentinel: &mut Sentinel,
    w_out: &Vec1D,
    b_out: f64,
    sigma_xi: f64,
    n_steps: usize,
    rng: &mut Rng,
) {
    let mut esn_tmp = esn.clone_esn();
    let inp = vec![1.0, 1.0];
    let mut x_sum = zeros_vec(esn.d);
    let mut a_sum = 0.0;
    for _ in 0..n_steps {
        esn_tmp.step(&inp, sigma_xi, rng);
        let a_body = body_readout(&esn_tmp.state, w_out, b_out);
        for i in 0..esn.d {
            x_sum[i] += esn_tmp.state[i];
        }
        a_sum += a_body;
    }
    let n = n_steps as f64;
    let x_mean: Vec1D = x_sum.iter().map(|v| v / n).collect();
    sentinel.set_baselines(x_mean, a_sum / n);
}

fn main() {
    let rho = 0.9;
    let sigma_xi = 0.15;
    let n_in = 2;

    let beta = 0.01;
    let alpha_oja = 1.0;
    let rho_min = 0.05;
    let rho_max = 0.99;
    let n_fast = 500;
    let n_habit = 200;

    let n_burnin = 500;
    let n_train = 2000;
    let n_warmup = 500;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];

    let dimensions = [5, 10, 20, 30, 50, 75];
    let defect_lengths: Vec<usize> = vec![10, 50, 100, 200, 500];
    let n_seeds = 20;

    println!("d,defect_length,seed,detection_time,alpha_min_reached,recovery_time,sentinel_payoff,tft_payoff");

    for &d in &dimensions {
        // Scale regularization with dimension
        let lambda_d = lambda_reg * d as f64 / 30.0;

        for seed_idx in 0..n_seeds {
            let seed = 2000 + seed_idx as u64 * 1000 + d as u64 * 100_000;
            let mut rng = Rng::new(seed);
            let mut esn = ESN::new(d, n_in, rho, &mut rng);

            // Development
            let mut esn_c = esn.clone_esn();
            let mut rng_c = Rng::new(seed + 2000);
            let coop_states = drive_and_collect(
                &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
            );
            let mut esn_d2 = esn.clone_esn();
            let mut rng_d2 = Rng::new(seed + 3000);
            let defect_states_data = drive_and_collect(
                &mut esn_d2, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d2,
            );
            let (w_out, b_out) = train_readout_ridge(
                &coop_states, &defect_states_data, target_c, target_d, lambda_d,
            );

            // Habituation
            for _ in 0..n_habit {
                let mut fast_states = Vec::with_capacity(n_fast);
                for _ in 0..n_fast {
                    esn.step(&coop_input, sigma_xi, &mut rng);
                    fast_states.push(esn.state.clone());
                }
                esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
            }

            for &defect_len in &defect_lengths {
                // Build schedule: Coop(500) → Defect(L) → Coop(500)
                let mut opp_schedule = Vec::new();
                for _ in 0..500 { opp_schedule.push(1.0); }
                for _ in 0..defect_len { opp_schedule.push(0.0); }
                for _ in 0..500 { opp_schedule.push(1.0); }
                let t_total = opp_schedule.len();
                let defect_start = 500;
                let defect_end = 500 + defect_len;

                // Dynamic sentinel
                let mut esn_eval = esn.clone_esn();
                let mut rng_eval = Rng::new(seed + 5000 + defect_len as u64);
                warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

                let mut sentinel = Sentinel::default_sentinel(d);
                let mut rng_bl = Rng::new(seed + 5500 + defect_len as u64);
                init_sentinel_baselines(&esn, &mut sentinel, &w_out, b_out, sigma_xi, 200, &mut rng_bl);

                let (alpha_trace, actions_sentinel) = run_coupled_dynamic(
                    &mut esn_eval, &mut sentinel, &w_out, b_out,
                    &opp_schedule, sigma_xi, &mut rng_eval,
                );

                // TfT baseline
                let mut esn_tft = esn.clone_esn();
                let mut rng_tft = Rng::new(seed + 6000 + defect_len as u64);
                warmup_cooperative(&mut esn_tft, n_warmup, sigma_xi, &mut rng_tft);

                let actions_tft = run_tft(
                    &mut esn_tft, &w_out, b_out,
                    &opp_schedule, sigma_xi, &mut rng_tft,
                );

                // Detection time
                let detection_time = {
                    let mut dt = defect_len;
                    for t in defect_start..defect_end {
                        if alpha_trace[t] < 0.75 {
                            dt = t - defect_start;
                            break;
                        }
                    }
                    dt
                };

                let alpha_min_reached = alpha_trace[defect_start..defect_end]
                    .iter().cloned().fold(f64::INFINITY, f64::min);

                let recovery_time = {
                    let mut rt = t_total - defect_end;
                    for t in defect_end..t_total {
                        if alpha_trace[t] > 0.7 {
                            rt = t - defect_end;
                            break;
                        }
                    }
                    rt
                };

                let sentinel_payoff: f64 = (0..t_total)
                    .map(|t| payoff(actions_sentinel[t], opp_schedule[t]))
                    .sum();
                let tft_payoff: f64 = (0..t_total)
                    .map(|t| payoff(actions_tft[t], opp_schedule[t]))
                    .sum();

                println!("{},{},{},{},{:.6},{},{:.4},{:.4}",
                    d, defect_len, seed_idx, detection_time, alpha_min_reached,
                    recovery_time, sentinel_payoff, tft_payoff);
            }
            eprintln!("  d={} seed {} done", d, seed_idx);
        }
        eprintln!("=== d={} complete ===", d);
    }
    eprintln!("Done.");
}

// ==========================================================================
// perturbation.rs — Environmental perturbation response
//
// Revised protocol:
//   Phase 1: Train readout on fresh reservoir (development)
//   Phase 2: Habituate with Oja only (no W_in adaptation)
//   Phase 3: Warm up from cooperative basin, then test perturbation
//
// Perturbation schedule:
//   T_pre (cooperative) → T_defect (opponent defects) → T_recover (cooperative)
//
// Output CSV: agent,seed,t,body_output,action,opp_action
// ==========================================================================

mod esn;
use esn::*;

fn run_coupled(
    esn: &mut ESN,
    alpha: f64,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>) {
    let n = opp_actions.len();
    let mut body_outs = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let a_body = body_readout(&esn.state, w_out, b_out);
        let a_cog = last_opp;
        let action = alpha * a_body + (1.0 - alpha) * a_cog;
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);
        body_outs.push(a_body);
        actions.push(action);
        last_opp = a_opp;
    }
    (body_outs, actions)
}

fn warmup_cooperative(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) {
    let inp = vec![1.0, 1.0];
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
    }
}

fn main() {
    let d = 30;
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

    let t_pre = 200;
    let t_defect = 100;
    let t_recover = 200;
    let t_total = t_pre + t_defect + t_recover;

    let mut opp_schedule = Vec::with_capacity(t_total);
    for _ in 0..t_pre { opp_schedule.push(1.0); }
    for _ in 0..t_defect { opp_schedule.push(0.0); }
    for _ in 0..t_recover { opp_schedule.push(1.0); }

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];
    let n_seeds = 20;

    println!("agent,seed,t,body_output,action,opp_action");

    for seed_idx in 0..n_seeds {
        let seed = 300 + seed_idx as u64 * 1000;
        let mut rng = Rng::new(seed);
        let mut esn = ESN::new(d, n_in, rho, &mut rng);

        // Phase 1: Development
        let mut esn_c = esn.clone_esn();
        let mut rng_c = Rng::new(seed + 2000);
        let coop_states = drive_and_collect(
            &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
        );
        let mut esn_d = esn.clone_esn();
        let mut rng_d = Rng::new(seed + 3000);
        let defect_states = drive_and_collect(
            &mut esn_d, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d,
        );
        let (w_out, b_out) = train_readout_ridge(
            &coop_states, &defect_states, target_c, target_d, lambda_reg,
        );

        // Phase 2: Habituation (Oja only)
        for _ in 0..n_habit {
            let mut fast_states = Vec::with_capacity(n_fast);
            for _ in 0..n_fast {
                esn.step(&coop_input, sigma_xi, &mut rng);
                fast_states.push(esn.state.clone());
            }
            esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
        }

        // Phase 3: Perturbation test

        // Agent 1: α=1 (body-governed)
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 5000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);
            let (body_outs, actions) = run_coupled(
                &mut esn_eval, 1.0, &w_out, b_out,
                &opp_schedule, sigma_xi, &mut rng_eval,
            );
            for t in 0..t_total {
                println!("alpha1,{},{},{:.6},{:.6},{:.1}",
                    seed_idx, t, body_outs[t], actions[t], opp_schedule[t]);
            }
        }

        // Agent 2: α=0 (cognitive TfT)
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 6000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);
            let (body_outs, actions) = run_coupled(
                &mut esn_eval, 0.0, &w_out, b_out,
                &opp_schedule, sigma_xi, &mut rng_eval,
            );
            for t in 0..t_total {
                println!("alpha0,{},{},{:.6},{:.6},{:.1}",
                    seed_idx, t, body_outs[t], actions[t], opp_schedule[t]);
            }
        }

        // Agent 3: α=0.5 (mixed)
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 7000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);
            let (body_outs, actions) = run_coupled(
                &mut esn_eval, 0.5, &w_out, b_out,
                &opp_schedule, sigma_xi, &mut rng_eval,
            );
            for t in 0..t_total {
                println!("alpha05,{},{},{:.6},{:.6},{:.1}",
                    seed_idx, t, body_outs[t], actions[t], opp_schedule[t]);
            }
        }

        // Agent 4: Dynamic sentinel
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 9000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let mut sentinel = Sentinel::default_sentinel(d);
            // Initialize baselines from cooperative warmup
            {
                let mut esn_bl = esn.clone_esn();
                let mut rng_bl = Rng::new(seed + 9500);
                let inp_bl = vec![1.0, 1.0];
                let n_bl = 200;
                let mut x_sum = zeros_vec(d);
                let mut a_sum = 0.0;
                for _ in 0..n_bl {
                    esn_bl.step(&inp_bl, sigma_xi, &mut rng_bl);
                    let a_b = body_readout(&esn_bl.state, &w_out, b_out);
                    for i in 0..d { x_sum[i] += esn_bl.state[i]; }
                    a_sum += a_b;
                }
                let x_mean: Vec1D = x_sum.iter().map(|v| v / n_bl as f64).collect();
                sentinel.set_baselines(x_mean, a_sum / n_bl as f64);
            }

            let mut last_opp_s = opp_schedule[0];
            for t in 0..t_total {
                let a_opp = opp_schedule[t];
                let a_body = body_readout(&esn_eval.state, &w_out, b_out);
                let a_cog = last_opp_s;
                sentinel.update(&esn_eval.state, a_body, a_cog);
                let action = sentinel.alpha * a_body + (1.0 - sentinel.alpha) * a_cog;
                let input = vec![action, a_opp];
                esn_eval.step(&input, sigma_xi, &mut rng_eval);
                println!("dynamic,{},{},{:.6},{:.6},{:.1}",
                    seed_idx, t, a_body, action, opp_schedule[t]);
                last_opp_s = a_opp;
            }
        }

        // Agent 5: AllC constant
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 8000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);
            for t in 0..t_total {
                let a_opp = opp_schedule[t];
                let body_out = body_readout(&esn_eval.state, &w_out, b_out);
                esn_eval.step(&vec![1.0, a_opp], sigma_xi, &mut rng_eval);
                println!("allc,{},{},{:.6},{:.6},{:.1}",
                    seed_idx, t, body_out, 1.0, a_opp);
            }
        }

        eprintln!("seed {} done", seed_idx);
    }
    eprintln!("Done.");
}

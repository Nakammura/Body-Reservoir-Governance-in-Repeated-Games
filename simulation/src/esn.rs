// ==========================================================================
// esn.rs — Echo State Network core library
//
// Implements:
//   - tanh-ESN with intrinsic noise (Definition 2.2 in paper)
//   - Oja habituation rule (Eq. 18 in paper)
//   - Strategy simulation (AllC, AllD, TfT)
//   - k-NN KL divergence estimator (Kraskov et al. 2004)
//   - Xoshiro256** RNG
// ==========================================================================
#![allow(dead_code)]

use std::f64::consts::PI;

// ---- RNG (xoshiro256**) ----
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(1); // avoid zero seed
        let mut s = [0u64; 4];
        for i in 0..4 {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            s[i] = z ^ (z >> 31);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    pub fn normal_vec(&mut self, n: usize, sigma: f64) -> Vec<f64> {
        (0..n).map(|_| self.normal() * sigma).collect()
    }
}

// ---- Linear algebra primitives ----
pub type Vec1D = Vec<f64>;
pub type Mat2D = Vec<Vec<f64>>;

pub fn zeros_vec(n: usize) -> Vec1D {
    vec![0.0; n]
}

pub fn zeros_mat(r: usize, c: usize) -> Mat2D {
    vec![vec![0.0; c]; r]
}

pub fn mat_vec_mul(m: &Mat2D, v: &Vec1D) -> Vec1D {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

pub fn vec_add(a: &Vec1D, b: &Vec1D) -> Vec1D {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

pub fn vec_scale(a: &Vec1D, s: f64) -> Vec1D {
    a.iter().map(|x| x * s).collect()
}

fn tanh_vec(v: &Vec1D) -> Vec1D {
    v.iter().map(|x| x.tanh()).collect()
}

// ---- Spectral radius via power iteration ----
pub fn spectral_radius(m: &Mat2D, rng: &mut Rng) -> f64 {
    let n = m.len();
    let mut v: Vec1D = (0..n).map(|_| rng.normal()).collect();
    let mut norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return 0.0;
    }
    v = vec_scale(&v, 1.0 / norm);

    for _ in 0..300 {
        let w = mat_vec_mul(m, &v);
        norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        v = vec_scale(&w, 1.0 / norm);
    }
    norm
}

pub fn rescale_to_rho(m: &mut Mat2D, target_rho: f64, rng: &mut Rng) {
    let current = spectral_radius(m, rng);
    if current < 1e-15 {
        return;
    }
    let scale = target_rho / current;
    for row in m.iter_mut() {
        for val in row.iter_mut() {
            *val *= scale;
        }
    }
}

// ---- ESN ----
pub struct ESN {
    pub d: usize,
    pub n_in: usize,
    pub w_res: Mat2D,
    pub w_in: Mat2D,
    pub bias: Vec1D,
    pub state: Vec1D,
    pub linear: bool,
}

impl ESN {
    /// Create a new ESN with given dimension, spectral radius, noise params
    pub fn new(d: usize, n_in: usize, rho: f64, rng: &mut Rng) -> Self {
        // Random Gaussian W_res, then rescale
        let mut w_res = zeros_mat(d, d);
        for i in 0..d {
            for j in 0..d {
                w_res[i][j] = rng.normal() / (d as f64).sqrt();
            }
        }
        rescale_to_rho(&mut w_res, rho, rng);

        // Input weights ~ N(0, 0.25I) => sigma=0.5
        let mut w_in = zeros_mat(d, n_in);
        for i in 0..d {
            for j in 0..n_in {
                w_in[i][j] = rng.normal() * 0.5;
            }
        }

        // Bias ~ N(0, 0.09I) => sigma=0.3
        let bias: Vec1D = (0..d).map(|_| rng.normal() * 0.3).collect();
        let state = zeros_vec(d);

        ESN {
            d,
            n_in,
            w_res,
            w_in,
            bias,
            state,
            linear: false,
        }
    }

    /// Create a linear ESN (f = identity, no bias) for validating linear-Gaussian theory
    pub fn new_linear(d: usize, n_in: usize, rho: f64, rng: &mut Rng) -> Self {
        let mut esn = Self::new(d, n_in, rho, rng);
        esn.linear = true;
        // Zero out bias for the linear case to match the linear-Gaussian theory
        esn.bias = zeros_vec(d);
        esn
    }

    /// One step of reservoir dynamics:
    ///   tanh-ESN: x(t+1) = tanh(W_res x(t) + W_in s(t) + b) + xi(t)
    ///   linear ESN: x(t+1) = W_res x(t) + W_in s(t) + xi(t)
    pub fn step(&mut self, input: &Vec1D, sigma_xi: f64, rng: &mut Rng) {
        let wx = mat_vec_mul(&self.w_res, &self.state);
        let ws = mat_vec_mul(&self.w_in, input);
        let pre: Vec1D = wx
            .iter()
            .zip(ws.iter())
            .zip(self.bias.iter())
            .map(|((a, b), c)| a + b + c)
            .collect();
        let post = if self.linear { pre } else { tanh_vec(&pre) };
        let noise = rng.normal_vec(self.d, sigma_xi);
        self.state = vec_add(&post, &noise);
    }

    /// Step with zero input (natural dynamics)
    pub fn step_zero(&mut self, sigma_xi: f64, rng: &mut Rng) {
        let zero_input = zeros_vec(self.n_in);
        self.step(&zero_input, sigma_xi, rng);
    }

    /// Oja habituation update (Eq. 18)
    pub fn oja_update(&mut self, states: &[Vec1D], beta: f64, alpha_oja: f64, rho_min: f64, rho_max: f64) {
        let n = states.len() as f64;
        for i in 0..self.d {
            for j in 0..self.d {
                let avg_xixj: f64 = states.iter().map(|s| s[i] * s[j]).sum::<f64>() / n;
                let avg_xi2: f64 = states.iter().map(|s| s[i] * s[i]).sum::<f64>() / n;
                let delta = beta * (avg_xixj - alpha_oja * avg_xi2 * self.w_res[i][j]);
                self.w_res[i][j] += delta;
            }
        }
        // Homeostatic spectral radius projection
        let mut rng_temp = Rng::new(42);
        let rho = spectral_radius(&self.w_res, &mut rng_temp);
        if rho > rho_max {
            rescale_to_rho(&mut self.w_res, rho_max, &mut rng_temp);
        } else if rho < rho_min && rho > 1e-10 {
            rescale_to_rho(&mut self.w_res, rho_min, &mut rng_temp);
        }
    }

    /// Clone the ESN (for sampling without disturbing habituation state)
    pub fn clone_esn(&self) -> Self {
        ESN {
            d: self.d,
            n_in: self.n_in,
            w_res: self.w_res.clone(),
            w_in: self.w_in.clone(),
            bias: self.bias.clone(),
            state: zeros_vec(self.d),
            linear: self.linear,
        }
    }
}

// ---- Dynamic Sentinel (v5: body-driven metacognitive governance) ----
//
// The sentinel models the metacognitive layer as a lightweight governance
// mechanism.  It does NOT detect anomalies itself; rather, the body reservoir
// generates a composite "discomfort" signal D(t) from three sources:
//   D_state:    reservoir state deviation from habituated baseline
//   D_output:   body readout deviation from habituated output
//   D_disagree: disagreement between body and cognitive outputs
// The sentinel adjusts α(t) based on this body-generated signal,
// filtered by a receptivity parameter ρ_rec.

pub struct Sentinel {
    pub alpha: f64,
    // Governance parameters (set by metacognition)
    pub alpha_0: f64,     // baseline trust level
    pub eta_up: f64,      // recovery rate (slow)
    pub eta_down: f64,    // intervention sharpness (fast)
    pub theta: f64,       // intervention threshold
    pub alpha_min: f64,   // floor on alpha
    // Composite signal weights
    pub w_state: f64,
    pub w_output: f64,
    pub w_disagree: f64,
    // Receptivity: how faithfully body signals are received
    pub rho_rec: f64,
    // Body baselines (EMA-tracked)
    pub x_baseline: Vec1D,
    pub a_baseline: f64,
    pub ema_rate: f64,
    // Last computed discomfort (for logging)
    pub last_d_state: f64,
    pub last_d_output: f64,
    pub last_d_disagree: f64,
    pub last_d_total: f64,
}

impl Sentinel {
    pub fn new(
        d: usize,
        alpha_0: f64,
        eta_up: f64,
        eta_down: f64,
        theta: f64,
        alpha_min: f64,
        w_state: f64,
        w_output: f64,
        w_disagree: f64,
        rho_rec: f64,
        ema_rate: f64,
    ) -> Self {
        Sentinel {
            alpha: alpha_0,
            alpha_0,
            eta_up,
            eta_down,
            theta,
            alpha_min,
            w_state,
            w_output,
            w_disagree,
            rho_rec,
            x_baseline: zeros_vec(d),
            a_baseline: 0.95,
            ema_rate,
            last_d_state: 0.0,
            last_d_output: 0.0,
            last_d_disagree: 0.0,
            last_d_total: 0.0,
        }
    }

    pub fn default_sentinel(d: usize) -> Self {
        Self::new(
            d,
            0.85,  // alpha_0
            0.05,  // eta_up
            0.5,   // eta_down
            0.1,   // theta
            0.05,  // alpha_min
            0.3,   // w_state
            0.3,   // w_output
            0.4,   // w_disagree
            1.0,   // rho_rec (unused, kept for API compat)
            0.02,  // ema_rate
        )
    }

    /// Initialize baselines from a warmup phase
    pub fn set_baselines(&mut self, x_baseline: Vec1D, a_baseline: f64) {
        self.x_baseline = x_baseline;
        self.a_baseline = a_baseline;
    }

    /// Update alpha(t) given body state, body output, and cognitive output.
    /// The body generates the discomfort signal; the sentinel adjusts alpha.
    pub fn update(&mut self, x: &Vec1D, a_body: f64, a_cog: f64) {
        let d = x.len();

        // Body-generated composite discomfort signal
        let d_state = {
            let sq_sum: f64 = x.iter().zip(&self.x_baseline)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            sq_sum.sqrt() / (d as f64).sqrt()
        };
        let d_output = (a_body - self.a_baseline).abs();
        let d_disagree = (a_body - a_cog).abs();

        let d_raw = self.w_state * d_state
            + self.w_output * d_output
            + self.w_disagree * d_disagree;

        // D_eff = D (no receptivity filter; ρ_rec removed for parsimony)
        let d_eff = d_raw;

        // Store for logging
        self.last_d_state = d_state;
        self.last_d_output = d_output;
        self.last_d_disagree = d_disagree;
        self.last_d_total = d_eff;

        // Update EMA baselines
        self.a_baseline = self.a_baseline * (1.0 - self.ema_rate)
            + a_body * self.ema_rate;
        for i in 0..d {
            self.x_baseline[i] = self.x_baseline[i] * (1.0 - self.ema_rate)
                + x[i] * self.ema_rate;
        }

        // Leaky integrator with sharp downward kicks
        let kick = if d_eff > self.theta {
            self.eta_down * (d_eff - self.theta)
        } else {
            0.0
        };
        self.alpha = self.alpha + self.eta_up * (self.alpha_0 - self.alpha) - kick;

        // Clamp
        if self.alpha < self.alpha_min {
            self.alpha = self.alpha_min;
        }
        if self.alpha > 1.0 {
            self.alpha = 1.0;
        }
    }
}

// ---- Input encoding ----
pub fn input_c(n_in: usize) -> Vec1D {
    let mut v = zeros_vec(n_in);
    v[0] = 1.0;
    v
}

pub fn input_d(n_in: usize) -> Vec1D {
    let mut v = zeros_vec(n_in);
    v[1] = 1.0;
    v
}

// ---- Strategy simulation ----

/// AllC: always receives cooperation input (in AllC population)
pub fn simulate_allc(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) -> Vec<Vec1D> {
    let inp = input_c(esn.n_in);
    let mut states = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
        states.push(esn.state.clone());
    }
    states
}

/// AllD: always receives defection input (opponent response to defection in AllD pop)
pub fn simulate_alld(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) -> Vec<Vec1D> {
    let inp = input_d(esn.n_in);
    let mut states = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
        states.push(esn.state.clone());
    }
    states
}

/// TfT in mixed population: with prob (1-eps) input=C, with prob eps input=D
pub fn simulate_tft(
    esn: &mut ESN,
    n_steps: usize,
    epsilon: f64,
    sigma_xi: f64,
    rng: &mut Rng,
) -> Vec<Vec1D> {
    let mut states = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let inp = if rng.next_f64() < epsilon {
            input_d(esn.n_in)
        } else {
            input_c(esn.n_in)
        };
        esn.step(&inp, sigma_xi, rng);
        states.push(esn.state.clone());
    }
    states
}

/// Zero-input simulation (natural distribution)
pub fn simulate_natural(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) -> Vec<Vec1D> {
    let mut states = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        esn.step_zero(sigma_xi, rng);
        states.push(esn.state.clone());
    }
    states
}

// ---- k-NN KL divergence estimator (Pérez-Cruz 2008) ----
// KL(P || Q) from samples p_samples ~ P, q_samples ~ Q
// Optimized: track only k smallest distances instead of full sort

fn kth_smallest_dist(point: &Vec1D, samples: &[Vec1D], k: usize, skip_idx: Option<usize>) -> f64 {
    // Maintain a max-heap of size k (using a sorted small vec)
    let mut top_k = vec![f64::INFINITY; k];

    for (j, sample) in samples.iter().enumerate() {
        if Some(j) == skip_idx {
            continue;
        }
        let dist_sq: f64 = point
            .iter()
            .zip(sample)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let dist = dist_sq.sqrt();

        // Insert into top-k if smaller than current max
        if dist < top_k[k - 1] {
            top_k[k - 1] = dist;
            // Bubble down to maintain sorted order
            let mut pos = k - 1;
            while pos > 0 && top_k[pos] < top_k[pos - 1] {
                top_k.swap(pos, pos - 1);
                pos -= 1;
            }
        }
    }
    top_k[k - 1]
}

pub fn kl_knn(p_samples: &[Vec1D], q_samples: &[Vec1D], k: usize) -> f64 {
    let n = p_samples.len();
    let m = q_samples.len();
    let d = p_samples[0].len();

    let mut sum = 0.0;
    for i in 0..n {
        let rho_k = kth_smallest_dist(&p_samples[i], p_samples, k, Some(i)).max(1e-300);
        let nu_k = kth_smallest_dist(&p_samples[i], q_samples, k, None).max(1e-300);
        sum += (nu_k / rho_k).ln();
    }

    (d as f64) / (n as f64) * sum + (m as f64 / (n as f64 - 1.0)).ln()
}

// ---- Gaussian KL divergence estimator ----
// Computes KL(P || Q) assuming both are approximately Gaussian.
// KL(N(mu_p, Sigma_p) || N(mu_q, Sigma_q))
//   = 1/2 [tr(Sigma_q^{-1} Sigma_p) - d + (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p) + ln(det Sigma_q / det Sigma_p)]

pub fn sample_mean(samples: &[Vec1D]) -> Vec1D {
    let n = samples.len() as f64;
    let d = samples[0].len();
    let mut mu = zeros_vec(d);
    for s in samples {
        for i in 0..d {
            mu[i] += s[i];
        }
    }
    for i in 0..d {
        mu[i] /= n;
    }
    mu
}

pub fn sample_covariance(samples: &[Vec1D], mu: &Vec1D) -> Mat2D {
    let n = samples.len() as f64;
    let d = mu.len();
    let mut cov = zeros_mat(d, d);
    for s in samples {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += (s[i] - mu[i]) * (s[j] - mu[j]);
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= n - 1.0;
        }
    }
    cov
}

/// Cholesky decomposition L such that A = L L^T. Returns L (lower triangular).
/// Returns None if not positive definite.
pub fn cholesky(a: &Mat2D) -> Option<Mat2D> {
    let d = a.len();
    let mut l = zeros_mat(d, d);
    for i in 0..d {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            if i == j {
                let val = a[i][i] - s;
                if val <= 0.0 { return None; }
                l[i][j] = val.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Log determinant from Cholesky factor: ln(det A) = 2 * sum(ln(L_ii))
pub fn log_det_from_cholesky(l: &Mat2D) -> f64 {
    2.0 * (0..l.len()).map(|i| l[i][i].ln()).sum::<f64>()
}

/// Solve L x = b where L is lower triangular (forward substitution)
pub fn solve_lower(l: &Mat2D, b: &Vec1D) -> Vec1D {
    let d = b.len();
    let mut x = zeros_vec(d);
    for i in 0..d {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve L^T x = b where L is lower triangular (back substitution)
pub fn solve_upper_t(l: &Mat2D, b: &Vec1D) -> Vec1D {
    let d = b.len();
    let mut x = zeros_vec(d);
    for i in (0..d).rev() {
        let mut s = b[i];
        for j in (i + 1)..d {
            s -= l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve A x = b where A = L L^T
pub fn solve_cholesky(l: &Mat2D, b: &Vec1D) -> Vec1D {
    let y = solve_lower(l, b);
    solve_upper_t(l, &y)
}

/// KL(P || Q) using Gaussian approximation from samples
pub fn kl_gaussian_from_samples(p_samples: &[Vec1D], q_samples: &[Vec1D]) -> f64 {
    let d = p_samples[0].len();

    let mu_p = sample_mean(p_samples);
    let mu_q = sample_mean(q_samples);
    let sig_p = sample_covariance(p_samples, &mu_p);
    let sig_q = sample_covariance(q_samples, &mu_q);

    // Add small regularization to ensure positive definiteness
    let eps = 1e-8;
    let mut sig_q_reg = sig_q.clone();
    for i in 0..d {
        sig_q_reg[i][i] += eps;
    }
    let mut sig_p_reg = sig_p.clone();
    for i in 0..d {
        sig_p_reg[i][i] += eps;
    }

    let l_q = match cholesky(&sig_q_reg) {
        Some(l) => l,
        None => return f64::NAN,
    };
    let l_p = match cholesky(&sig_p_reg) {
        Some(l) => l,
        None => return f64::NAN,
    };

    // ln(det Sigma_q / det Sigma_p)
    let log_det_q = log_det_from_cholesky(&l_q);
    let log_det_p = log_det_from_cholesky(&l_p);
    let log_det_ratio = log_det_q - log_det_p;

    // tr(Sigma_q^{-1} Sigma_p): solve Sigma_q X = Sigma_p column by column
    let mut trace = 0.0;
    for j in 0..d {
        let col_p: Vec1D = (0..d).map(|i| sig_p_reg[i][j]).collect();
        let sol = solve_cholesky(&l_q, &col_p);
        trace += sol[j]; // diagonal element of Sigma_q^{-1} Sigma_p
    }

    // (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p)
    let diff: Vec1D = mu_p.iter().zip(&mu_q).map(|(a, b)| a - b).collect();
    let sol_diff = solve_cholesky(&l_q, &diff);
    let mahal: f64 = diff.iter().zip(&sol_diff).map(|(a, b)| a * b).sum();

    0.5 * (trace - d as f64 + mahal + log_det_ratio)
}

// ---- v4 coupled dynamics utilities ----

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

pub fn dot_product(a: &Vec1D, b: &Vec1D) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Body readout: σ(w_out · x + b_out) → [0,1]
pub fn body_readout(state: &Vec1D, w_out: &Vec1D, b_out: f64) -> f64 {
    sigmoid(dot_product(state, w_out) + b_out)
}

/// Train readout via ridge regression in logit-space.
/// Maps cooperation-compatible states → target_c (e.g. 0.95)
/// and defection-compatible states → target_d (e.g. 0.05).
pub fn train_readout_ridge(
    coop_states: &[Vec1D],
    defect_states: &[Vec1D],
    target_c: f64,
    target_d: f64,
    lambda_reg: f64,
) -> (Vec1D, f64) {
    let d = coop_states[0].len();
    let d_aug = d + 1;
    let y_c = logit(target_c);
    let y_d = logit(target_d);

    let mut xtx = zeros_mat(d_aug, d_aug);
    let mut xty = zeros_vec(d_aug);

    for s in coop_states {
        for i in 0..d {
            for j in 0..d {
                xtx[i][j] += s[i] * s[j];
            }
            xtx[i][d] += s[i];
            xtx[d][i] += s[i];
            xty[i] += s[i] * y_c;
        }
        xtx[d][d] += 1.0;
        xty[d] += y_c;
    }

    for s in defect_states {
        for i in 0..d {
            for j in 0..d {
                xtx[i][j] += s[i] * s[j];
            }
            xtx[i][d] += s[i];
            xtx[d][i] += s[i];
            xty[i] += s[i] * y_d;
        }
        xtx[d][d] += 1.0;
        xty[d] += y_d;
    }

    let n_total = (coop_states.len() + defect_states.len()) as f64;
    for i in 0..d {
        xtx[i][i] += lambda_reg * n_total;
    }

    let l = cholesky(&xtx).expect("Cholesky failed in readout training");
    let w_aug = solve_cholesky(&l, &xty);

    (w_aug[..d].to_vec(), w_aug[d])
}

/// Run reservoir with arbitrary constant input, collect states after burn-in.
pub fn drive_and_collect(
    esn: &mut ESN,
    input: &Vec1D,
    n_burnin: usize,
    n_collect: usize,
    sigma_xi: f64,
    rng: &mut Rng,
) -> Vec<Vec1D> {
    for _ in 0..n_burnin {
        esn.step(input, sigma_xi, rng);
    }
    let mut states = Vec::with_capacity(n_collect);
    for _ in 0..n_collect {
        esn.step(input, sigma_xi, rng);
        states.push(esn.state.clone());
    }
    states
}

/// Generic habituation step with arbitrary input.
pub fn habituation_step_generic(
    esn: &mut ESN,
    input: &Vec1D,
    n_fast: usize,
    sigma_xi: f64,
    beta: f64,
    alpha_oja: f64,
    rho_min: f64,
    rho_max: f64,
    eta_in: f64,
    rng: &mut Rng,
) {
    let mut fast_states = Vec::with_capacity(n_fast);
    for _ in 0..n_fast {
        esn.step(input, sigma_xi, rng);
        fast_states.push(esn.state.clone());
    }
    esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);

    if eta_in > 0.0 {
        let mu_driven = sample_mean(&fast_states);
        let mut esn_nat = esn.clone_esn();
        let mut natural_states = Vec::with_capacity(n_fast);
        for _ in 0..n_fast {
            esn_nat.step_zero(sigma_xi, rng);
            natural_states.push(esn_nat.state.clone());
        }
        let mu_natural = sample_mean(&natural_states);
        for i in 0..esn.d {
            let delta_i = mu_driven[i] - mu_natural[i];
            for j in 0..esn.n_in {
                esn.w_in[i][j] -= eta_in * delta_i * input[j];
            }
        }
    }
}

// ---- Habituation loop: run fast dynamics, collect states, do Oja update ----
pub fn habituation_step(
    esn: &mut ESN,
    n_fast: usize,
    sigma_xi: f64,
    beta: f64,
    alpha_oja: f64,
    rho_min: f64,
    rho_max: f64,
    rng: &mut Rng,
) {
    let inp = input_c(esn.n_in); // AllC habituation
    let mut fast_states = Vec::with_capacity(n_fast);
    for _ in 0..n_fast {
        esn.step(&inp, sigma_xi, rng);
        fast_states.push(esn.state.clone());
    }
    esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
}

/// Habituation with W_in sensory adaptation.
///
/// In addition to the Oja rule on W_res, this updates W_in to reduce the
/// mean-shift between driven and natural dynamics:
///
///   ΔW_in[i][j] = -η_in · (μ_driven[i] - μ_natural[i]) · s̄[j]
///
/// For AllC (s̄ = [1,0]), only W_in[:,0] (the C-input channel) is reduced,
/// preserving sensitivity to novel inputs (e.g., defection).
/// This models biological sensory adaptation: prolonged constant stimulation
/// shifts the baseline, reducing ongoing metabolic cost.
pub fn habituation_step_with_win_adapt(
    esn: &mut ESN,
    n_fast: usize,
    sigma_xi: f64,
    beta: f64,
    alpha_oja: f64,
    rho_min: f64,
    rho_max: f64,
    eta_in: f64,
    rng: &mut Rng,
) {
    let inp = input_c(esn.n_in); // AllC habituation

    // 1. Run fast dynamics under driven input, collect states
    let mut fast_states = Vec::with_capacity(n_fast);
    for _ in 0..n_fast {
        esn.step(&inp, sigma_xi, rng);
        fast_states.push(esn.state.clone());
    }

    // 2. Oja update on W_res (same as standard habituation)
    esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);

    // 3. W_in sensory adaptation: reduce mean-shift
    if eta_in > 0.0 {
        // Mean of driven states
        let mu_driven = sample_mean(&fast_states);

        // Collect natural states (same number of steps)
        let mut esn_nat = esn.clone_esn();
        let mut natural_states = Vec::with_capacity(n_fast);
        for _ in 0..n_fast {
            esn_nat.step_zero(sigma_xi, rng);
            natural_states.push(esn_nat.state.clone());
        }
        let mu_natural = sample_mean(&natural_states);

        // Gradient step: ΔW_in[i][j] = -η_in · (μ_driven[i] - μ_natural[i]) · s̄[j]
        for i in 0..esn.d {
            let delta_i = mu_driven[i] - mu_natural[i];
            for j in 0..esn.n_in {
                esn.w_in[i][j] -= eta_in * delta_i * inp[j];
            }
        }
    }
}

/// Estimate KL(q_strategy || p_T) at current habituation state
/// Uses Gaussian approximation (sample mean + covariance → analytical KL)
pub fn estimate_kl(
    esn: &ESN,
    strategy: &str,
    epsilon: f64,
    sigma_xi: f64,
    n_samples: usize,
    n_burnin: usize,
    _knn_k: usize,
    rng_seed: u64,
) -> f64 {
    // Sample from natural distribution p_T
    let mut esn_nat = esn.clone_esn();
    let mut rng_nat = Rng::new(rng_seed);
    let _ = simulate_natural(&mut esn_nat, n_burnin, sigma_xi, &mut rng_nat);
    let p_samples = simulate_natural(&mut esn_nat, n_samples, sigma_xi, &mut rng_nat);

    // Sample from driven distribution q(x | strategy)
    let mut esn_drv = esn.clone_esn();
    let mut rng_drv = Rng::new(rng_seed.wrapping_add(100_000));
    let q_samples = match strategy {
        "AllC" => {
            let _ = simulate_allc(&mut esn_drv, n_burnin, sigma_xi, &mut rng_drv);
            simulate_allc(&mut esn_drv, n_samples, sigma_xi, &mut rng_drv)
        }
        "TfT" => {
            let _ = simulate_tft(&mut esn_drv, n_burnin, epsilon, sigma_xi, &mut rng_drv);
            simulate_tft(&mut esn_drv, n_samples, epsilon, sigma_xi, &mut rng_drv)
        }
        "AllD" => {
            let _ = simulate_alld(&mut esn_drv, n_burnin, sigma_xi, &mut rng_drv);
            simulate_alld(&mut esn_drv, n_samples, sigma_xi, &mut rng_drv)
        }
        _ => panic!("Unknown strategy: {}", strategy),
    };

    kl_gaussian_from_samples(&q_samples, &p_samples).max(0.0)
}

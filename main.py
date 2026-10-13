"""
Pytom Canvas Love
Karya generatif: hati putih bercahaya dari partikel di atas latar hitam pekat.
Intro : benang kusut acak-acakan yang perlahan terurai lalu membentuk hati.
Detail: firework warna-warni di samping hati, riak detak, cincin orbit,
        bintang jatuh, bara naik, dan vignette halus.

Install : pip install pygame numpy
Jalankan: python pytom_canvas_love.py

Kontrol : mouse = dorong partikel | klik = gelombang partikel
          R / Spasi = ulangi animasi formasi | F / F11 = fullscreen | Esc / Q = keluar
"""
import math
import sys

import numpy as np
import pygame

TAU = math.tau
FPS = 60
BEAT_PERIOD = 1.9          # detik per detak (lub-dub)
MOUSE_RADIUS = 120         # jangkauan dorongan kursor (px)
WAVE_SPEED, WAVE_LIFE = 420.0, 1.8
RIPPLE_LIFE = 1.8                      # umur riak detak (detik)
STAR_SPEED, STAR_LIFE = 800.0, 0.8     # bintang jatuh
FW_KINDS = ("peony", "willow", "ring", "heart")
FW_WEIGHTS = (0.40, 0.25, 0.20, 0.15)

# fase intro: benang kusut -> hati
N_THREADS = 18
THREAD_K = np.array([1.0] * 4 + list(np.linspace(0.92, 0.10, 14)))  # skala kontur tiap benang
FORM_START, FORM_U, FORM_DUR = 1.6, 2.2, 2.4   # mulai terurai, sebaran sepanjang benang, lama transisi
T_BEAT, T_RV, T_RV2, T_FX = 6.3, 6.0, 7.0, 8.5  # detak, kontur, konstelasi, efek tambahan mulai muncul

# warna firework (RGB 0..1)
COLORS = ((0.30, 0.58, 1.00), (1.00, 0.27, 0.31), (1.00, 0.85, 0.30), (0.35, 1.00, 0.60),
          (1.00, 0.35, 0.78), (1.00, 0.60, 0.24), (0.30, 0.90, 1.00), (0.67, 0.43, 1.00))
HEART_COLORS = (COLORS[1], COLORS[4], (1.00, 0.50, 0.60))


# ----------------------------------------------------------------- util
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gauss(x, c, w):
    return math.exp(-(((x - c) / w) ** 2))


def hx(t, k):  # kurva hati parametrik, dipusatkan pada bounding box
    return 16.0 * np.sin(t) ** 3 * k


def hy(t, k):
    return -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t) + 2.5) * k


def tint(c, v):
    """Warna c (RGB 0..1) dengan kecerahan v (0..1) -> tuple int untuk pygame."""
    return tuple(int(max(0, min(255, 255 * v * x))) for x in c)


def grey(v):
    v = int(max(0, min(255, v)))
    return (v, v, v)


# ----------------------------------------------------------------- app
class Love:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pytom Canvas Love")
        self.win_size = (1100, 800)
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.win_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()
        self.W = self.H = None
        self._setup(*self.screen.get_size())
        self._build()

    # --- ukuran layar / layout
    def _setup(self, w, h):
        ow, oh = self.W, self.H
        self.W, self.H = w, h
        self.U = min(w / 32 * 0.6, h / 29 * 0.56)  # satuan hati, menyisakan ruang kosong
        self.cx, self.cy = w / 2, h / 2
        self.frame = pygame.Surface((w, h))
        self.lay = pygame.Surface((w, h))
        self.vig = self._make_vignette(w, h)
        if ow:  # sesuaikan posisi debu bintang saat resize
            self.dx *= w / ow
            self.dy *= h / oh

    # --- partikel
    def _build(self):
        r, W, H = self.rng, self.W, self.H
        sc = 0.6 if W * H < 520_000 else 1.0
        n_in, n_rim, n_flow = int(1700 * sc), int(850 * sc), int(240 * sc)

        # bagian dalam hati: sebaran merata lewat kontur yang diskalakan
        t = r.uniform(0, TAU, n_in)
        k = np.sqrt(r.uniform(0, 1, n_in))
        ux1, uy1 = hx(t, k), hy(t, k)
        u1 = t / TAU
        j1 = 4 + np.abs(k[:, None] - THREAD_K[None, 4:]).argmin(axis=1)
        tone1 = r.choice([1.0, 0.82, 0.62], n_in, p=[0.55, 0.30, 0.15])
        a1 = (0.22 + 0.5 * r.uniform(0, 1, n_in)) * (0.55 + 0.45 * k)
        s1 = r.uniform(0.5, 1.2, n_in)

        # siluet: partikel rapat di tepi
        t = r.uniform(0, TAU, n_rim)
        k = 1 - r.uniform(0, 1, n_rim) ** 3 * 0.05
        ux2 = hx(t, k) + r.uniform(-0.14, 0.14, n_rim)
        uy2 = hy(t, k) + r.uniform(-0.14, 0.14, n_rim)
        u2 = t / TAU
        j2 = r.integers(0, 4, n_rim)
        tone2 = r.choice([1.0, 0.82], n_rim, p=[0.7, 0.3])
        a2 = r.uniform(0.55, 0.95, n_rim)
        s2 = r.uniform(0.7, 1.5, n_rim)

        # partikel yang mengalir mengikuti kontur
        self.ft = r.uniform(0, TAU, n_flow)
        self.fk = np.array([1.0, 1.0, 0.93, 0.86])[np.arange(n_flow) % 4]
        self.fsp = np.where(r.uniform(0, 1, n_flow) < 0.78, 1, -1) * r.uniform(0.16, 0.4, n_flow)
        ux3, uy3 = hx(self.ft, self.fk), hy(self.ft, self.fk)
        u3 = self.ft / TAU
        j3 = np.where(self.fk >= 0.99, r.integers(0, 4, n_flow),
                      4 + np.abs(self.fk[:, None] - THREAD_K[None, 4:]).argmin(axis=1))
        a3 = r.uniform(0.7, 1.0, n_flow)
        s3 = r.uniform(1.0, 1.9, n_flow)

        self.ns = n_in + n_rim  # jumlah partikel statis
        self.n_in = n_in
        self.n = self.ns + n_flow
        self.ux = np.concatenate([ux1, ux2, ux3])
        self.uy = np.concatenate([uy1, uy2, uy3])
        self.tone = np.concatenate([tone1, tone2, np.ones(n_flow)])
        self.alpha = np.concatenate([a1, a2, a3])
        self.size = np.concatenate([s1, s2, s3])
        # tiap partikel menempel pada satu benang (j) di posisi u; benang terurai dari ujung ke ujung
        self.pj = np.concatenate([j1, j2, j3])
        self.pu = np.concatenate([u1, u2, u3]) % 1.0
        self.dl = 0.08 * self.pj + FORM_U * self.pu + r.uniform(0, 0.3, self.n)
        # parameter gumpalan benang kusut: 3 gelombang sinus per sumbu, frekuensi bulat -> kurva tertutup
        self.cf = r.integers(1, 6, (N_THREADS, 2, 3)).astype(float)
        self.cp = r.uniform(0, TAU, (N_THREADS, 2, 3))
        self.cs = r.uniform(-0.6, 0.6, (N_THREADS, 2, 3))
        self.co = r.uniform(-0.12, 0.12, (N_THREADS, 2))
        self.cam = r.uniform(0.55, 1.0, (N_THREADS, 2))
        self.ph = r.uniform(0, TAU, self.n)
        self.ts = r.uniform(0.4, 1.5, self.n)

        # konstelasi: garis tipis antar titik terpilih
        self.nodes = r.integers(0, self.ns, 84)
        pts = np.stack([self.ux[self.nodes], self.uy[self.nodes]], 1)
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        seen = set()
        self.pairs = []
        for i in range(len(pts)):
            for j in np.argsort(D[i])[1:4]:
                if D[i, j] < 5.4 and (min(i, j), max(i, j)) not in seen:
                    seen.add((min(i, j), max(i, j)))
                    self.pairs.append((self.nodes[i], self.nodes[j]))
        self.nph = r.uniform(0, TAU, len(self.nodes))

        # debu bintang latar dengan kedalaman
        nd = 130 if W * H < 520_000 else 230
        self.dz = r.uniform(0, 1, nd) ** 1.6
        self.dx, self.dy = r.uniform(0, W, nd), r.uniform(0, H, nd)
        self.dvx = r.uniform(-3, 3, nd) * (0.3 + self.dz)
        self.dvy = r.uniform(-5, -1, nd) * (0.3 + self.dz)
        self.dox, self.doy = np.zeros(nd), np.zeros(nd)
        self.dph, self.dts = r.uniform(0, TAU, nd), r.uniform(0.3, 1.3, nd)
        self.par = np.zeros(2)

        self.replay()

    def replay(self):
        """Mulai dari gumpalan benang kusut, lalu biarkan terurai membentuk hati."""
        self.px, self.py = self._chaos(self.pj, self.pu, 0.0)
        self.vx, self.vy = np.zeros(self.n), np.zeros(self.n)
        self.e = np.zeros(self.n)
        self.t = 0.0
        self.waves, self.bursts, self.sparks = [], [], []
        self.next_spark = T_FX - 0.5
        # detail tambahan
        self.ripples, self.rockets, self.fwp, self.fw_booms, self.embers = [], [], [], [], []
        self.flashes = []
        self.star, self.next_star = None, T_FX + 2.5
        self.next_fw, self.fw_side, self.prev_p = T_FX + 0.5, 1, 0.0

    # --- simulasi
    def burst(self, x, y):
        self.waves.append([x, y, 0.0])
        self.waves = self.waves[-4:]
        for _ in range(70):
            a, s = self.rng.uniform(0, TAU), self.rng.uniform(120, 380)
            self.bursts.append([x, y, math.cos(a) * s, math.sin(a) * s, 0.0, self.rng.uniform(0.9, 1.8)])

    def update(self, dt, bs, tick=False):
        t = self.t
        # partikel flow: maju sepanjang kontur
        self.ft += self.fsp * dt
        self.ux[self.ns:] = hx(self.ft, self.fk)
        self.uy[self.ns:] = hy(self.ft, self.fk)

        e = self.e = smooth((t - FORM_START - self.dl) / FORM_DUR)  # 0 = benang kusut, 1 = hati
        cxp, cyp = self._chaos(self.pj, self.pu, t)
        sw = np.sin(np.pi * e) * 40  # lilitan kecil saat benang terurai
        tx = (cxp * (1 - e) + (self.cx + self.ux * self.U * bs + np.sin(t * 0.7 + self.ph) * 0.9) * e
              + sw * np.sin(self.ph + t * 1.3))
        ty = (cyp * (1 - e) + (self.cy + self.uy * self.U * bs + np.cos(t * 0.6 + self.ph) * 0.9) * e
              + sw * np.cos(self.ph + t * 1.1))
        ax = (tx - self.px) * 10
        ay = (ty - self.py) * 10

        if pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            dx, dy = self.px - mx, self.py - my
            d = np.sqrt(dx * dx + dy * dy) + 1e-6
            f = np.clip(1 - d / MOUSE_RADIUS, 0, None) ** 2 * 1000
            ax += dx / d * f
            ay += dy / d * f
            self.mouse = (mx, my)
        else:
            self.mouse = None

        for w in self.waves:
            dx, dy = self.px - w[0], self.py - w[1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-6
            band = np.clip(1 - np.abs(d - w[2] * WAVE_SPEED) / 55, 0, None)
            f = band * (1 - w[2] / WAVE_LIFE) * 2600
            ax += dx / d * f
            ay += dy / d * f
            w[2] += dt
        self.waves = [w for w in self.waves if w[2] < WAVE_LIFE]

        dm = math.exp(-4.2 * dt)
        self.vx = (self.vx + ax * dt) * dm
        self.vy = (self.vy + ay * dt) * dm
        self.px += self.vx * dt
        self.py += self.vy * dt

        # debu bintang
        W, H = self.W, self.H
        self.dx = (self.dx + self.dvx * dt) % W
        self.dy = (self.dy + self.dvy * dt) % H
        tgt = np.zeros(2)
        if self.mouse:
            tgt = np.array([(self.mouse[0] - self.cx) * -0.02, (self.mouse[1] - self.cy) * -0.02])
            ddx = self.dx + self.dox - self.mouse[0]
            ddy = self.dy + self.doy - self.mouse[1]
            dd = np.sqrt(ddx * ddx + ddy * ddy) + 1e-6
            f = np.clip(1 - dd / 100, 0, None) ** 2 * 90 * dt
            self.dox += ddx / dd * f
            self.doy += ddy / dd * f
        self.par += (tgt - self.par) * min(1.0, dt * 2)
        self.dox *= math.exp(-1.5 * dt)
        self.doy *= math.exp(-1.5 * dt)

        # percikan klik
        for b in self.bursts:
            k = math.exp(-1.6 * dt)
            b[2] *= k
            b[3] *= k
            b[0] += b[2] * dt
            b[1] += b[3] * dt
            b[4] += dt
        self.bursts = [b for b in self.bursts if b[4] < b[5]]

        # sparkle acak
        if t > self.next_spark and len(self.sparks) < 14:
            self.next_spark = t + self.rng.uniform(0.15, 0.5)
            self.sparks.append([int(self.rng.integers(0, self.ns)), 0.0,
                                self.rng.uniform(0.9, 1.6), self.rng.uniform(5, 11)])
        for s in self.sparks:
            s[1] += dt
        self.sparks = [s for s in self.sparks if s[1] < s[2]]

        self._update_extras(dt, tick)

    # --- render
    def draw(self, b, bs, rv, rv2):
        W, H, t, U = self.W, self.H, self.t, self.U
        intro = float(smooth(t / 0.8))

        # 1) partikel (anti-aliased, akumulasi bilinear ke buffer kecerahan)
        tw = 0.62 + 0.38 * np.sin(t * self.ts + self.ph)
        a = np.minimum(1, self.alpha * self.tone * tw * (0.92 + 0.22 * b) * (0.5 + 0.5 * self.e) * intro)
        wt = [a * (0.7 + 0.8 * self.size)]
        X, Y = [self.px], [self.py]

        sz = 0.4 + self.dz * 1.1
        da = (0.1 + 0.5 * self.dz) * (0.55 + 0.45 * np.sin(t * self.dts + self.dph))
        X.append(self.dx + self.dox + self.par[0] * self.dz)
        Y.append(self.dy + self.doy + self.par[1] * self.dz)
        wt.append(da * (0.7 + 0.8 * sz))

        if self.bursts:
            bu = np.array(self.bursts)
            X.append(bu[:, 0])
            Y.append(bu[:, 1])
            wt.append((1 - bu[:, 4] / bu[:, 5]) ** 1.5 * 0.9)

        ex = self._extra_points()
        if ex:
            X.append(ex[0])
            Y.append(ex[1])
            wt.append(ex[2])

        X, Y, Wt = np.concatenate(X), np.concatenate(Y), np.concatenate(wt)
        x0, y0 = np.floor(X).astype(np.int32), np.floor(Y).astype(np.int32)
        fx, fy = X - x0, Y - y0
        idx, wts = [], []
        for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + ox, y0 + oy
            m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            idx.append(xi[m] * H + yi[m])
            wts.append((Wt * w)[m])
        buf = np.bincount(np.concatenate(idx), np.concatenate(wts), minlength=W * H).reshape(W, H)
        g = (np.clip(buf, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
        cp = self._color_points()
        if cp:
            self._splat_color(rgb, *cp)  # percikan firework berwarna
        pygame.surfarray.blit_array(self.frame, rgb)

        # 2) garis: kontur berlapis, siluet, konstelasi, sparkle, ring gelombang
        lay = self.lay
        lay.fill((0, 0, 0))
        cx, cy = self.cx, self.cy

        def contour(k, n):
            tt = np.linspace(0, TAU, n)
            return np.column_stack((cx + hx(tt, k) * U * bs, cy + hy(tt, k) * U * bs)).tolist()

        self._draw_threads(lay, intro)
        for i, k in enumerate((0.86, 0.72, 0.58, 0.44, 0.31, 0.19)):
            kk = k * (1 + 0.006 * math.sin(t * 0.8 + i))
            v = 255 * (0.12 - i * 0.012) * rv * (0.8 + 0.5 * b)
            pygame.draw.aalines(lay, grey(v), True, contour(kk, 110))
        pygame.draw.aalines(lay, grey(255 * (0.5 + 0.25 * b) * rv), True, contour(1.0, 200))

        px, py, maxd = self.px, self.py, U * 6.5
        lc = grey(255 * (0.13 + 0.1 * b) * rv2)
        for i, j in self.pairs:
            if math.hypot(px[i] - px[j], py[i] - py[j]) < maxd:
                pygame.draw.aaline(lay, lc, (px[i], py[i]), (px[j], py[j]))
        for n, ph in zip(self.nodes, self.nph):
            v = (0.35 + 0.35 * math.sin(t * 0.8 + ph)) * rv2
            pygame.draw.circle(lay, grey(255 * v * 0.18), (px[n], py[n]), 5)
            pygame.draw.circle(lay, grey(255 * (v + 0.2 * rv2)), (px[n], py[n]), 1)

        for i in range(self.ns, self.n):  # ekor pendek partikel flow
            pygame.draw.line(lay, grey(255 * 0.3 * rv), (px[i] - self.vx[i] * 0.2, py[i] - self.vy[i] * 0.2),
                             (px[i], py[i]))

        for n, l, m, r in self.sparks:
            v = math.sin(math.pi * l / m)
            x, y, rr = px[n], py[n], r * v
            pygame.draw.circle(lay, grey(255 * 0.05 * v), (x, y), max(1, int(rr * 1.1)))
            pygame.draw.circle(lay, grey(255 * 0.14 * v), (x, y), max(1, int(rr * 0.45)))
            pygame.draw.circle(lay, grey(255 * 0.9 * v), (x, y), max(1, int(rr * 0.14)))
            c, q = grey(255 * 0.75 * v), rr * 0.55
            pygame.draw.line(lay, c, (x - rr, y), (x + rr, y))
            pygame.draw.line(lay, c, (x, y - rr), (x, y + rr))
            c = grey(255 * 0.35 * v)
            pygame.draw.line(lay, c, (x - q, y - q), (x + q, y + q))
            pygame.draw.line(lay, c, (x - q, y + q), (x + q, y - q))

        for wx, wy, wt_ in self.waves:
            r = int(wt_ * WAVE_SPEED)
            if r > 1:
                pygame.draw.circle(lay, grey(255 * 0.22 * (1 - wt_ / WAVE_LIFE)), (wx, wy), r, 1)

        self._draw_extras(lay, b, rv, rv2, contour)
        self.frame.blit(lay, (0, 0), special_flags=pygame.BLEND_ADD)

        # 3) glow berlapis (bloom), berdenyut mengikuti detak
        s1 = pygame.transform.smoothscale(self.frame, (max(1, W // 4), max(1, H // 4)))
        s2 = pygame.transform.smoothscale(s1, (max(1, W // 12), max(1, H // 12)))
        gate = 0.55 + 0.45 * rv
        for s, gain in ((s1, (1.9 + 0.9 * b) * gate), (s2, (3.2 + 1.8 * b) * gate)):
            arr = pygame.surfarray.array3d(s).astype(np.float32) * gain
            pygame.surfarray.blit_array(s, np.minimum(arr, 255).astype(np.uint8))
            self.frame.blit(pygame.transform.smoothscale(s, (W, H)), (0, 0), special_flags=pygame.BLEND_ADD)

        self.frame.blit(self.vig, (0, 0), special_flags=pygame.BLEND_RGB_MULT)  # vignette
        self.screen.blit(self.frame, (0, 0))
        pygame.display.flip()

    # --- intro benang kusut
    def _chaos(self, j, u, t):
        """Posisi titik pada benang j (parameter u 0..1) di gumpalan benang yang menggeliat."""
        a = TAU * u[:, None, None] * self.cf[j] + self.cp[j] + t * self.cs[j]  # (n, sumbu, gelombang)
        sv = (np.sin(a) * np.array([0.55, 0.30, 0.15])).sum(-1)               # (n, sumbu)
        x = self.cx + self.co[j, 0] * self.W + self.cam[j, 0] * 0.38 * self.W * sv[:, 0]
        y = self.cy + self.co[j, 1] * self.H + self.cam[j, 1] * 0.38 * self.H * sv[:, 1]
        return x, y

    def _draw_threads(self, lay, intro):
        """Garis benang tipis: kusut di awal, lalu perlahan menjadi kontur hati dan memudar."""
        t, M = self.t, 120
        j = np.repeat(np.arange(N_THREADS), M)
        u = np.tile(np.linspace(0, 1, M), N_THREADS)
        e = smooth((t - FORM_START - (0.08 * j + FORM_U * u)) / FORM_DUR)
        cxp, cyp = self._chaos(j, u, t)
        tt = u * TAU
        X = (cxp * (1 - e) + (self.cx + hx(tt, THREAD_K[j]) * self.U) * e).reshape(N_THREADS, M)
        Y = (cyp * (1 - e) + (self.cy + hy(tt, THREAD_K[j]) * self.U) * e).reshape(N_THREADS, M)
        em = e.reshape(N_THREADS, M).mean(1)
        for i in range(N_THREADS):
            v = 255 * 0.30 * (1 - em[i]) ** 0.8 * intro
            if v >= 2:
                pygame.draw.aalines(lay, grey(v), False, np.column_stack((X[i], Y[i])).tolist())

    # --- firework berwarna
    def _color_points(self):
        if not self.fwp:
            return None
        return (np.array([s[0] for s in self.fwp]), np.array([s[1] for s in self.fwp]),
                np.array([self._spark_alpha(s) * 1.8 for s in self.fwp]),
                np.array([s[10] for s in self.fwp]) * 255)

    def _splat_color(self, rgb, X, Y, Wt, C):
        W, H = self.W, self.H
        x0, y0 = np.floor(X).astype(np.int64), np.floor(Y).astype(np.int64)
        fx, fy = X - x0, Y - y0
        ids, vals = [], []
        for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + ox, y0 + oy
            m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            ids.append(xi[m] * H + yi[m])
            vals.append(((Wt * w)[m])[:, None] * C[m])
        ids, vals = np.concatenate(ids), np.concatenate(vals)
        if not len(ids):
            return
        uniq, inv = np.unique(ids, return_inverse=True)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, vals)
        ux_, uy_ = uniq // H, uniq % H
        rgb[ux_, uy_] = np.minimum(255, rgb[ux_, uy_].astype(np.float32) + sums).astype(np.uint8)

    # --- detail tambahan -------------------------------------------------
    def _make_vignette(self, w, h):
        xs = (np.arange(w) - w / 2) / (w / 2)
        ys = (np.arange(h) - h / 2) / (h / 2)
        r = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)
        g = ((1 - 0.55 * smooth((r - 0.6) / 0.85)) * 255).astype(np.uint8)
        v = pygame.Surface((w, h))
        pygame.surfarray.blit_array(v, np.stack([g, g, g], axis=-1))
        return v

    def _fw_zone(self):
        """Titik ledakan di ruang kosong samping hati (atau atas/bawah di layar sempit)."""
        W, H, U, rng = self.W, self.H, self.U, self.rng
        self.fw_side *= -1
        free_x = self.cx - 16 * U * 1.06
        if free_x >= 90:
            zx = self.cx + self.fw_side * (16 * U * 1.06 + free_x / 2)
            zy = self.cy + rng.uniform(-0.28, 0.05) * H
            R = min(free_x * 0.42, H * 0.2, 150)
            return zx, zy, R, zx + rng.uniform(-0.03, 0.03) * W, H + 10
        free_y = self.cy - 14.5 * U * 1.06
        if free_y < 50:
            return None
        zx = self.cx + rng.uniform(-0.25, 0.25) * W
        zy = free_y * 0.5 if self.fw_side > 0 else H - free_y * 0.5
        return zx, zy, min(W * 0.22, free_y * 0.42, 120), zx, H + 10

    def _explode(self, x, y, kind, R, cols):
        r = self.rng
        self.flashes.append([x, y, 0.0, R, cols[0]])
        if kind == "willow":
            n, m, g, drag = 30, 2.1, 130.0, 1.7
            sp = R * 1.6 * r.uniform(0.5, 1.0, n)
            ang = r.uniform(0, TAU, n)
            vx, vy = np.cos(ang) * sp, np.sin(ang) * sp
        elif kind == "ring":
            n, m, g, drag = 40, 1.4, 30.0, 2.2
            ang = np.arange(n) * TAU / n + r.uniform(-0.03, 0.03, n)
            vx, vy = np.cos(ang) * R * 2.2, np.sin(ang) * R * 2.2
        elif kind == "heart":  # ledakan mini berbentuk hati
            n, m, g, drag = 40, 1.5, 20.0, 2.2
            tt = np.arange(n) * TAU / n
            vx, vy = hx(tt, 1.0) / 17 * R * 2.2, hy(tt, 1.0) / 17 * R * 2.2
        else:  # peony
            n, m, g, drag = 36, 1.4, 40.0, 2.2
            sp = R * 2.2 * r.uniform(0.8, 1.0, n)
            ang = r.uniform(0, TAU, n)
            vx, vy = np.cos(ang) * sp, np.sin(ang) * sp
        for i in range(n):
            self.fwp.append([x, y, float(vx[i]), float(vy[i]), 0.0, m, g, drag,
                             r.uniform(0.55, 0.85), r.uniform(0, TAU), cols[i % len(cols)]])

    @staticmethod
    def _spark_alpha(s):
        f = s[4] / s[5]
        a = s[8] * (1 - f) ** 1.4
        return a * (0.6 + 0.4 * math.sin(s[4] * 38 + s[9])) if f > 0.6 else a  # bara berkelip di akhir

    def _update_extras(self, dt, tick):
        t, rng, W, H = self.t, self.rng, self.W, self.H
        self.fw_booms = [x for x in self.fw_booms if t - x < 1.5]
        self.ripples = [x for x in self.ripples if t - x < RIPPLE_LIFE]

        if tick and t > T_FX:
            self.ripples.append(t + 0.19)  # riak muncul tepat di puncak "lub"
            for _ in range(int(rng.integers(5, 10))):  # bara naik dari tepi hati
                if len(self.embers) < 40:
                    i = int(rng.integers(self.n_in, self.ns))
                    self.embers.append([self.px[i], self.py[i], rng.uniform(-14, 14),
                                        rng.uniform(-60, -28), 0.0, rng.uniform(2.0, 3.2), rng.uniform(0, TAU)])
            if t >= self.next_fw and len(self.rockets) + len(self.fw_booms) < 2:
                z = self._fw_zone()
                if z:
                    zx, zy, R, x0, y0 = z
                    kind = str(rng.choice(FW_KINDS, p=FW_WEIGHTS))
                    if kind == "heart":
                        cols = [HEART_COLORS[int(rng.integers(0, 3))]]
                    else:  # satu warna, atau dua warna berpadu
                        ci = rng.choice(len(COLORS), 2, replace=False)
                        cols = [COLORS[ci[0]]] if rng.uniform() < 0.6 else [COLORS[ci[0]], COLORS[ci[1]]]
                    self.rockets.append([x0, y0, zx, zy, 0.0, 0.8 + 0.5 * abs(y0 - zy) / H, kind, R, [], cols])
                    self.next_fw = t + rng.uniform(2.5, 4.0)

        alive = []
        for rk in self.rockets:
            rk[4] += dt
            p = min(1.0, rk[4] / rk[5])
            e = 1 - (1 - p) ** 2
            x = rk[0] + (rk[2] - rk[0]) * e + math.sin(p * 9) * 3 * (1 - p)
            y = rk[1] + (rk[3] - rk[1]) * e
            rk[8].append((x, y))
            del rk[8][:-14]
            if p >= 1.0:
                self._explode(x, y, rk[6], rk[7], rk[9])
                self.fw_booms.append(t)
            else:
                alive.append(rk)
        self.rockets = alive

        for s in self.fwp:
            k = math.exp(-s[7] * dt)
            s[2] *= k
            s[3] = (s[3] + s[6] * dt) * k
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            s[4] += dt
        self.fwp = [s for s in self.fwp if s[4] < s[5]]
        for f in self.flashes:
            f[2] += dt
        self.flashes = [f for f in self.flashes if f[2] < 0.35]

        for em in self.embers:
            em[4] += dt
            em[2] += math.sin(em[4] * 3 + em[6]) * 10 * dt
            em[0] += em[2] * dt
            em[1] += em[3] * dt
        self.embers = [em for em in self.embers if em[4] < em[5]]

        if self.star is None:
            if t > T_FX + 2 and t >= self.next_star:
                ang = rng.uniform(0.35, 0.6)
                sgn = 1 if rng.uniform() < 0.5 else -1
                self.star = [rng.uniform(0.15, 0.85) * W, rng.uniform(0.02, 0.22) * H,
                             sgn * math.cos(ang), math.sin(ang), 0.0]
        else:
            s = self.star
            s[4] += dt
            s[0] += s[2] * STAR_SPEED * dt
            s[1] += s[3] * STAR_SPEED * dt
            if s[4] > STAR_LIFE:
                self.star, self.next_star = None, t + rng.uniform(10, 15)

    def _extra_points(self):
        xs, ys, ws = [], [], []
        for e in self.embers:
            f = e[4] / e[5]
            a = 0.85 * (1 - f) ** 1.3 * (0.7 + 0.3 * math.sin(e[4] * 6 + e[6])) * min(1.0, e[4] / 0.15)
            xs.append(e[0]); ys.append(e[1]); ws.append(a * 1.3)
        if not xs:
            return None
        return np.array(xs), np.array(ys), np.array(ws)

    def _draw_extras(self, lay, b, rv, rv2, contour):
        t = self.t
        # riak detak: kontur hati yang memuai lalu memudar
        for t0 in self.ripples:
            pr = (t - t0) / RIPPLE_LIFE
            if 0 <= pr < 1:
                k = 1.0 + 0.55 * (1 - (1 - pr) ** 2)
                pygame.draw.aalines(lay, grey(255 * 0.2 * (1 - pr) ** 1.5 * rv), True, contour(k, 140))

        self._draw_orbit(lay, b, rv2)

        # firework: garis peluncur tipis + ekor percikan
        for rk in self.rockets:
            tr = rk[8]
            for i in range(len(tr) - 1):
                pygame.draw.line(lay, tint(rk[9][0], 0.45 * i / len(tr)), tr[i], tr[i + 1])
            if tr:
                pygame.draw.circle(lay, grey(200), tr[-1], 1)
        for s in self.fwp:
            c = tint(s[10], self._spark_alpha(s) * 0.6)
            pygame.draw.line(lay, c, (s[0] - s[2] * 0.06, s[1] - s[3] * 0.06), (s[0], s[1]))

        for fx, fy, fl, fR, fc in self.flashes:  # kilatan warna saat meledak
            pr = fl / 0.35
            pygame.draw.circle(lay, tint(fc, 0.22 * (1 - pr)), (fx, fy), int(fR * 0.5 * math.sqrt(pr)) + 2)

        # bintang jatuh
        if self.star:
            x, y, dx, dy, l = self.star
            fade = math.sin(math.pi * min(l, STAR_LIFE) / STAR_LIFE) ** 0.7
            L = 140
            for i in range(10):
                pygame.draw.aaline(lay, grey(255 * 0.8 * (1 - i / 10) * fade),
                                   (x - dx * L * i / 10, y - dy * L * i / 10),
                                   (x - dx * L * (i + 1) / 10, y - dy * L * (i + 1) / 10))
            pygame.draw.circle(lay, grey(255 * fade), (x, y), 1)

    def _draw_orbit(self, lay, b, rv2):
        """Dua cincin putus-putus yang berputar pelan + titik cahaya berekor komet."""
        if rv2 < 0.01:
            return
        t, U = self.t, self.U
        lim = min(self.W, self.H) * 0.46
        rings = ((min(20.5 * U, lim * 0.85), 1.0, 0.0, 0.05, 96, 0.10, 0.35),
                 (min(24.5 * U, lim), 0.86, -0.35, -0.035, 120, 0.07, -0.28))
        for rad, ratio, tilt, dash_sp, nd, lum, dot_sp in rings:
            rad *= 1 + 0.012 * b
            ct, st = math.cos(tilt), math.sin(tilt)

            def pt(a):
                ex, ey = rad * math.cos(a), rad * ratio * math.sin(a)
                return self.cx + ex * ct - ey * st, self.cy + ex * st + ey * ct

            col, step = grey(255 * lum * rv2), TAU / nd
            for i in range(nd):
                a0 = i * step + t * dash_sp
                pygame.draw.aaline(lay, col, pt(a0), pt(a0 + step * 0.5))
            sg = 1 if dot_sp > 0 else -1
            for off in ((0.0, math.pi) if ratio == 1.0 else (0.0,)):
                a = t * dot_sp + off
                for q in range(10):  # ekor komet
                    pygame.draw.aaline(lay, grey(255 * 0.7 * rv2 * (1 - q / 10)),
                                       pt(a - sg * q * 0.03), pt(a - sg * (q + 1) * 0.03))
                hp = pt(a)
                pygame.draw.circle(lay, grey(255 * 0.12 * rv2), hp, 5)
                pygame.draw.circle(lay, grey(255 * 0.9 * rv2), hp, 2)

    # --- jendela
    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags, size = (pygame.FULLSCREEN, (0, 0)) if self.fullscreen else (pygame.RESIZABLE, self.win_size)
        self.screen = pygame.display.set_mode(size, flags)
        self._setup(*self.screen.get_size())

    def run(self):
        while True:
            dt = min(0.033, self.clock.tick(FPS) / 1000)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if ev.key in (pygame.K_f, pygame.K_F11):
                        self.toggle_fullscreen()
                    if ev.key in (pygame.K_r, pygame.K_SPACE):
                        self.replay()
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.burst(*ev.pos)
                elif ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.win_size = (ev.w, ev.h)
                    self.screen = pygame.display.get_surface()
                    self._setup(ev.w, ev.h)

            self.t += dt
            p = (self.t % BEAT_PERIOD) / BEAT_PERIOD
            tick = p < self.prev_p  # awal detak baru
            self.prev_p = p
            b = (gauss(p, 0.1, 0.06) + 0.65 * gauss(p, 0.3, 0.07)) * float(smooth((self.t - T_BEAT) / 2.5))
            bs = 1 + 0.04 * b + 0.006 * math.sin(self.t * 0.5)
            rv = float(smooth((self.t - T_RV) / 2.4))
            rv2 = float(smooth((self.t - T_RV2) / 2.2))
            self.update(dt, bs, tick)
            self.draw(b, bs, rv, rv2)


if __name__ == "__main__":
    Love().run()
    pygame.quit()
    sys.exit()
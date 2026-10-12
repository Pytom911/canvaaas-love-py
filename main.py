"""
Pytom Canvas Love
Karya generatif: hati putih bercahaya dari partikel di atas latar hitam pekat.

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
        tone1 = r.choice([1.0, 0.82, 0.62], n_in, p=[0.55, 0.30, 0.15])
        a1 = (0.22 + 0.5 * r.uniform(0, 1, n_in)) * (0.55 + 0.45 * k)
        s1 = r.uniform(0.5, 1.2, n_in)

        # siluet: partikel rapat di tepi
        t = r.uniform(0, TAU, n_rim)
        k = 1 - r.uniform(0, 1, n_rim) ** 3 * 0.05
        ux2 = hx(t, k) + r.uniform(-0.14, 0.14, n_rim)
        uy2 = hy(t, k) + r.uniform(-0.14, 0.14, n_rim)
        tone2 = r.choice([1.0, 0.82], n_rim, p=[0.7, 0.3])
        a2 = r.uniform(0.55, 0.95, n_rim)
        s2 = r.uniform(0.7, 1.5, n_rim)

        # partikel yang mengalir mengikuti kontur
        self.ft = r.uniform(0, TAU, n_flow)
        self.fk = np.array([1.0, 1.0, 0.93, 0.86])[np.arange(n_flow) % 4]
        self.fsp = np.where(r.uniform(0, 1, n_flow) < 0.78, 1, -1) * r.uniform(0.16, 0.4, n_flow)
        ux3, uy3 = hx(self.ft, self.fk), hy(self.ft, self.fk)
        a3 = r.uniform(0.7, 1.0, n_flow)
        s3 = r.uniform(1.0, 1.9, n_flow)

        self.ns = n_in + n_rim  # jumlah partikel statis
        self.n = self.ns + n_flow
        self.ux = np.concatenate([ux1, ux2, ux3])
        self.uy = np.concatenate([uy1, uy2, uy3])
        self.tone = np.concatenate([tone1, tone2, np.ones(n_flow)])
        self.alpha = np.concatenate([a1, a2, a3])
        self.size = np.concatenate([s1, s2, s3])
        self.dl = np.concatenate([r.uniform(0, 2.8, self.ns), r.uniform(1.5, 3.4, n_flow)])
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
        """Sebar partikel lalu biarkan berkumpul membentuk hati."""
        r = self.rng
        ang = r.uniform(0, TAU, self.n)
        dist = r.uniform(0.2, 0.9, self.n) * max(self.W, self.H)
        self.px = self.cx + np.cos(ang) * dist
        self.py = self.cy + np.sin(ang) * dist
        self.vx, self.vy = r.uniform(-14, 14, self.n), r.uniform(-14, 14, self.n)
        self.e = np.zeros(self.n)
        self.t = 0.0
        self.waves, self.bursts, self.sparks = [], [], []
        self.next_spark = 3.0

    # --- simulasi
    def burst(self, x, y):
        self.waves.append([x, y, 0.0])
        self.waves = self.waves[-4:]
        for _ in range(70):
            a, s = self.rng.uniform(0, TAU), self.rng.uniform(120, 380)
            self.bursts.append([x, y, math.cos(a) * s, math.sin(a) * s, 0.0, self.rng.uniform(0.9, 1.8)])

    def update(self, dt, bs):
        t = self.t
        # partikel flow: maju sepanjang kontur
        self.ft += self.fsp * dt
        self.ux[self.ns:] = hx(self.ft, self.fk)
        self.uy[self.ns:] = hy(self.ft, self.fk)

        e = self.e = smooth((t - self.dl) / 1.7)
        tx = self.cx + self.ux * self.U * bs + np.sin(t * 0.7 + self.ph) * 0.9 * e
        ty = self.cy + self.uy * self.U * bs + np.cos(t * 0.6 + self.ph) * 0.9 * e
        ax = (tx - self.px) * (10 * e + 0.015)
        ay = (ty - self.py) * (10 * e + 0.015)

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

        self.vx = (self.vx + ax * dt) * np.exp(-(0.4 + 3.8 * e) * dt)
        self.vy = (self.vy + ay * dt) * np.exp(-(0.4 + 3.8 * e) * dt)
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

    # --- render
    def draw(self, b, bs, rv, rv2):
        W, H, t, U = self.W, self.H, self.t, self.U

        # 1) partikel (anti-aliased, akumulasi bilinear ke buffer kecerahan)
        tw = 0.62 + 0.38 * np.sin(t * self.ts + self.ph)
        a = np.minimum(1, self.alpha * self.tone * tw * (0.92 + 0.22 * b) * (0.3 + 0.7 * self.e))
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
        pygame.surfarray.blit_array(self.frame, np.stack([g, g, g], axis=-1))

        # 2) garis: kontur berlapis, siluet, konstelasi, sparkle, ring gelombang
        lay = self.lay
        lay.fill((0, 0, 0))
        cx, cy = self.cx, self.cy

        def contour(k, n):
            tt = np.linspace(0, TAU, n)
            return np.column_stack((cx + hx(tt, k) * U * bs, cy + hy(tt, k) * U * bs)).tolist()

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

        self.frame.blit(lay, (0, 0), special_flags=pygame.BLEND_ADD)

        # 3) glow berlapis (bloom), berdenyut mengikuti detak
        s1 = pygame.transform.smoothscale(self.frame, (max(1, W // 4), max(1, H // 4)))
        s2 = pygame.transform.smoothscale(s1, (max(1, W // 12), max(1, H // 12)))
        gate = 0.4 + 0.6 * rv
        for s, gain in ((s1, (1.9 + 0.9 * b) * gate), (s2, (3.2 + 1.8 * b) * gate)):
            arr = pygame.surfarray.array3d(s).astype(np.float32) * gain
            pygame.surfarray.blit_array(s, np.minimum(arr, 255).astype(np.uint8))
            self.frame.blit(pygame.transform.smoothscale(s, (W, H)), (0, 0), special_flags=pygame.BLEND_ADD)

        self.screen.blit(self.frame, (0, 0))
        pygame.display.flip()

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
            b = (gauss(p, 0.1, 0.06) + 0.65 * gauss(p, 0.3, 0.07)) * float(smooth(self.t / 2.5))
            bs = 1 + 0.04 * b + 0.006 * math.sin(self.t * 0.5)
            rv = float(smooth((self.t - 2.4) / 3))
            rv2 = float(smooth((self.t - 3.2) / 2.5))
            self.update(dt, bs)
            self.draw(b, bs, rv, rv2)


if __name__ == "__main__":
    Love().run()
    pygame.quit()
    sys.exit()

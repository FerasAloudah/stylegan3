import cv2
import numpy as np

import dnnlib
from visualizer import AsyncRenderer


class LatentWalk:
    def __init__(self, pkl=None, anim=True):
        self.args = dnnlib.EasyDict(pkl=pkl)
        self.result = dnnlib.EasyDict()
        self.latent = dnnlib.EasyDict(x=0, y=0, anim=anim, speed=0.25)
        self.step_y = 100
        self._async_renderer = AsyncRenderer()

    def generate(self, dx=None, dy=None):
        self.drag(dx, dy)
        self.walk()
        if self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None and 'image' in result:
                return result.image

        return None

    def drag(self, dx, dy):
        if dx:
            self.latent.x += dx / 24 * 4e-2
        if dy:
            self.latent.y += dy / 24 * 4e-2

    def walk(self):
        seed = round(self.latent.x) + round(self.latent.y) * self.step_y
        self.latent.x = seed
        self.latent.y = 0
        frac_x = self.latent.x - round(self.latent.x)
        frac_y = self.latent.y - round(self.latent.y)
        self.latent.x += 0.01 - frac_x
        self.latent.y += 0.01 - frac_y
        self.latent.x += 1 * self.latent.speed  # replace 1 with time delta (curr_time - start_time)

        self.args.w0_seeds = []
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(self.latent.x) + ofs_x
            seed_y = np.floor(self.latent.y) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            if weight > 0:
                self.args.w0_seeds.append([seed, weight])


def main():
    latent_walk = LatentWalk(pkl='')
    while True:
        image = latent_walk.generate()
        cv2.imshow("image", image)
        cv2.waitKey()


if __name__ == '__main__':
    main()

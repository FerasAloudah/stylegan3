import cv2 as cv
from viz.renderer import Renderer
from visualizer import AsyncRenderer


class LatentWalk:
    def __init__(self, pkl=None, anim=True):
        self.args            = dnnlib.EasyDict(w0_seeds=[], pkl=pkl)
        self.result          = dnnlib.EasyDict()
        self.latent          = dnnlib.EasyDict(x=0, y=0, anim=anim, speed=0.25)
        self.step_y          = 100
        self._async_renderer = AsyncRenderer()

    def generate(self, dx, dy):
        self.args = dnnlib.EasyDict()
        self.drag(dx, dy)
        self.walk()
        if self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None and 'image' in result:
                return result

        return None

    def drag(self, dx, dy):
        self.latent.x += dx / 24 * 4e-2
        self.latent.y += dy / 24 * 4e-2

    def walk(self):
        seed = round(self.latent.x) + round(self.latent.y) * self.step_y
        self.latent.x = seed
        self.latent.y = 0
        frac_x = self.latent.x - round(self.latent.x)
        frac_y = self.latent.y - round(self.latent.y)
        self.latent.x += 0.01 - frac_x
        self.latent.y += 0.01 - frac_y
        self.latent.x += 1 * self.latent.speed # replace 1 with time delta (curr_time - start_time)

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
        img = latent_walk.generate()
        img = numpy.zeros([5, 5, 3])
        img[:, :, 0] = numpy.ones([5, 5]) * 64 / 255.0
        img[:, :, 1] = numpy.ones([5, 5]) * 128 / 255.0
        img[:, :, 2] = numpy.ones([5, 5]) * 192 / 255.0
        cv2.imshow("image", img)
        cv.waitKey()

if __name__ == '__main__':
    main()
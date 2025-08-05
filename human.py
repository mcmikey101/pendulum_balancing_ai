import pygame 
from collections import namedtuple
import math

pygame.init()
font = pygame.font.SysFont('arial', 25)

Vector = namedtuple('Vector', 'x, y')

class Point():
    def __init__(self, x, y, vec_x, vec_y, force_vec_x, force_vec_y, fixed=False, tethered=False):
        self.x = x
        self.y = y
        self.tethered = tethered
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.force_vec_x = force_vec_x
        self.force_vec_y = force_vec_y
        self.fixed = fixed

    def move(self, force, w, h, gravity_vec, link=None, rad=0):
        self.force_vec_x, self.force_vec_y = force
        self.vec_x += self.force_vec_x + gravity_vec.x
        self.vec_y += self.force_vec_y + gravity_vec.y

        self.x += self.vec_x
        self.y += self.vec_y

        if self.tethered:

            dx = self.x - link.x
            dy = self.y - link.y
            dist = math.hypot(dx, dy) + 1e-9
            correction = (rad - dist) / dist
            self.x += dx * correction
            self.y += dy * correction

            ux, uy  = dx/dist, dy/dist

            k = 1
            c = 1     
            dlen = dist - rad

            Fsx, Fsy = -k * dlen * ux, -k * dlen * uy
            vrel_x = self.vec_x - force.x
            vrel_y = self.vec_y - force.y
            Fdamp_x = -c * (vrel_x*ux + vrel_y*uy) * ux
            Fdamp_y = -c * (vrel_x*ux + vrel_y*uy) * uy

            self.vec_x += (Fsx + Fdamp_x)
            self.vec_y += (Fsy + Fdamp_y)

        if self.y >= h / 2 and self.fixed:
            self.y = h / 2
            self.vec_y = 0
        if self.y >= h - 30:
            self.y = h - 30
            self.vec_y = 0
        if self.y <= 30:
            self.y = 30
            self.vec_y = 0
        if self.x >= w - 90:
            self.x = w - 90
            self.vec_x = 0
        if self.x <= 90:
            self.x = 90
            self.vec_x = 0

        # if self.vec_x > 50:
        #     self.vec_x = 50
        # elif self.vec_x < -50:
        #     self.vec_x = -50
        # if self.vec_y > 50:
        #     self.vec_y = 50
        # elif self.vec_y < -50:
        #     self.vec_y = -50

class Pendulum():
    def __init__(self, w=1200, h=640, speed=60, gravity=10):
        self.w = w
        self.h = h
        self.stem_to_joint = 150
        self.joint_to_leaf = 100
        self.gravity = gravity
        self.speed = speed
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Pendulum')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.force = Vector(0, 0)
        self.stem = Point(self.w / 2, self.h / 2, 0, 0, 0, 0, fixed=True)
        self.joint = Point(self.w / 2, self.h / 2, 0, 0, 0, 0, tethered=True)
        self.leaf = Point(self.w / 2, self.h / 2, 0, 0, 0, 0, tethered=True)
        self.gravity_vec = Vector(0, self.gravity)
        self.time_up = 0
        self.frame_iteration = 0

    def sim_step(self):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.force = Vector(-2.5, 0)
                elif event.key == pygame.K_RIGHT:
                    self.force = Vector(2.5, 0)

        self.stem.move(self.force, self.w, self.h, self.gravity_vec)
        self.joint.move(Vector(-self.stem.vec_x * 0.8, self.stem.vec_y * 0.8), self.w, self.h, self.gravity_vec, link=Vector(self.stem.x, self.stem.y), rad=self.stem_to_joint)
        self.leaf.move(Vector(-self.joint.vec_x * 0.8, self.joint.vec_y * 0.8), self.w, self.h, self.gravity_vec, link=Vector(self.joint.x, self.joint.y), rad=self.joint_to_leaf)

        self.reward = 0
        self.done = False

        if self.leaf.y < self.h / 2:
            self.reward = 5
            self.time_up += 1
        else:
            self.reward = -1
        
        if self.frame_iteration == 50:
            self.done = True
            return self.reward, self.done, self.time_up
        
        self._update_ui()
        self.clock.tick(self.speed)

        return self.reward, self.done, self.time_up
    
    def _update_ui(self):
        self.display.fill((255, 255, 255))

        pygame.draw.line(self.display, (0, 0, 0), (self.stem.x, self.stem.y), (self.joint.x, self.joint.y), 10)
        pygame.draw.line(self.display, (0, 0, 0), (self.joint.x, self.joint.y), (self.leaf.x, self.leaf.y), 10)

        pygame.draw.circle(self.display, (0, 0, 255), (self.stem.x, self.stem.y), 30)
        pygame.draw.circle(self.display, (0, 255, 0), (self.joint.x, self.joint.y), 20)
        pygame.draw.circle(self.display, (255, 0, 0), (self.leaf.x, self.leaf.y), 20)

        text = font.render("Reward: " + str(self.reward) + " Time up: " + str(self.time_up) + "fvx: " + str(self.stem.force_vec_x) + "fvy: " + str(self.stem.force_vec_y), True, (0, 0, 0))
        self.display.blit(text, [0, 0])

        pygame.display.flip()


if __name__ == "__main__":
    sim = Pendulum()

    while True:
        reward, done, time_up = sim.sim_step()

    pygame.quit()
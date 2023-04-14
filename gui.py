import pygame
import sys
import random
import time
import math
# import testRecorded

##### CONFIG ###########
pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = pygame.display.get_desktop_sizes()[0]

BAR_COLOR = pygame.Color(0, 128, 255)  # blue
BAR_WIDTH = WINDOW_WIDTH * 0.9 * 0.05
BAR_HEIGHT = WINDOW_HEIGHT * 0.9 * 0.89
BAR_X = WINDOW_WIDTH * 0.9 * 0.05
BAR_Y = WINDOW_HEIGHT * 0.9 * 0.05

NOTE_COLOR = pygame.Color(0, 255, 162)  # green
NOTE_WIDTH = 20
NOTE_HEIGHT = 20
NOTE_START_X = WINDOW_WIDTH + 1

CURSOR_COLOR = pygame.Color(255, 255, 255)  # white
CURSOR_WIDTH = BAR_WIDTH + 30
CURSOR_HEIGHT = 150
CURSOR_START_Y = BAR_Y + BAR_HEIGHT - CURSOR_HEIGHT

#### END OF CONFIG #####


# class Cursor:
#     def __init__(self) -> None:
#         self.obj = pygame.Rect(BAR_X-15, CURSOR_START_Y,
#                                CURSOR_WIDTH, CURSOR_HEIGHT)

#     def move(self):
#         key = pygame.key.get_pressed()
#         dist = 1
#         if key[pygame.K_UP] and self.obj.top > BAR_Y:
#             self.obj = self.obj.move(0, -dist)
#         if key[pygame.K_DOWN] and self.obj.top + self.obj.height < BAR_Y + BAR_HEIGHT:
#             self.obj = self.obj.move(0, dist)
#         if key[pygame.K_q]:
#             sys.exit()

#     def eeg_move(self, voltage):
#         self.obj = self.obj.move(0, voltage*1e18)
#         if self.obj.top < BAR_Y:
#             self.obj.top = BAR_Y
#         if self.obj.top + self.obj.height > BAR_Y + BAR_HEIGHT:
#             self.obj.top = BAR_Y + BAR_HEIGHT - self.obj.height

#     def get_obj(self):
#         return self.obj

#     def get_x(self):
#         return self.obj.left

#     def get_y(self):
#         return self.obj.top


class Note:

    def __init__(self, pycircle: pygame.draw.circle, step_size=310) -> None:
        self.pycircle = pycircle
        self.cursor_touch = False
        self.frame_counter = 1
        self.step_size = step_size

    def shrink(self):
        self.pycircle.height -= self.step_size
        
    def get_radius(self):
        return self.pycircle.height//2

    def get_pycircle(self) -> pygame.Rect:
        return self.pycircle

    def get_x(self) -> int:
        return self.pycircle.left

    def get_y(self) -> int:
        return self.pycircle.top

    # unused anymore
    # def is_touching_center(self, cursor_x, cursor_y):
    #     if cursor_x <= self.get_x() and self.get_x() <= (cursor_x + CURSOR_WIDTH):
    #         if self.get_y() >= cursor_y and self.get_y() <= cursor_y + CURSOR_HEIGHT:
    #             return True
    #     return False


class RhythmGame:

    unit_time = 250
    notes_distance = 1000
    min_notes = 10
    max_notes = 20

    def __init__(self) -> None:
        self.window_width = 0.9 * WINDOW_WIDTH
        self.window_height = 0.9 * WINDOW_HEIGHT

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE)

        # self.cursor = Cursor()
        self.notes = []

    def add_note(self):

        pycircle = pygame.draw.circle(self.screen, NOTE_COLOR,
                                    (self.screen.get_width()//2, self.screen.get_height()//2), self.screen.get_height()//2, 5)
        
        note = Note(pycircle)
        self.notes.append(note)

    def update_screen(self) -> None:
        self.screen.fill((0, 0, 0))  # black
        # draw cursor
        self.draw_cursor()
        # draw cursor
        # self.cursor.move()
        # self.draw_cursor(self.cursor.get_obj())
        # draw all notes to the current frame
        # self.handle_notes()
        for note in self.notes:
            # get the note's obj(rect)
            self.draw_note(note.get_pycircle())

        # # handle notes
        # for note in self.notes:
        #     note.move()
        #     if note.is_touching_cursor(self.cursor.get_x(), self.cursor.get_y()) and not note.cursor_touch:  # hit
        #         self.add_hit_effect(note)
        #         note.cursor_touch = True
        #     if note.get_x() < 0:  # miss
        #         self.notes.remove(note)
        #         note.cursor_touch = False
        # if (len(self.notes) == 0):
        #     self.add_note()

        # render hit effects
        i = 0
        while i < len(self.notes):
            note = self.notes[i]
            radius = note.get_pycircle().height//2
            
            # make the hit effect discrete
            note.frame_counter += 1
            if (note.frame_counter % 360 == 0):
                note.shrink()
                note.frame_counter = 1
            
            
            if radius > 30:
                thickness = 2 - 1//(radius)
                if thickness < 0:
                    thickness = 0
                # effect = pygame.draw.circle(self.screen, NOTE_COLOR,
                #                             (effect.centerx, effect.centery),
                #                             radius, thickness)
                self.draw_note(note.get_pycircle())
                
                self.notes[i] = note
            else:
                self.notes.remove(note)
                i -= 1

            # if effect.top <= 1 or effect.left <= 1:
            #     self.notes.remove(effect)
            #     i -= 1
            i += 1

    def display(self) -> None:
        pygame.display.flip()

    def draw_cursor(self) -> None:
        note_color = pygame.Color(0, 255-100, 162-100)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//2, self.screen.get_height()//2), 70, 0)
        
        note_color = pygame.Color(0, 255-50, 162-50)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//2, self.screen.get_height()//2), 60, 0)
        
        note_color = pygame.Color(0, 255, 162)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//2, self.screen.get_height()//2), 50, 0)

    def draw_note(self, pycircle) -> None:
        pygame.draw.circle(self.screen, NOTE_COLOR,
                                    (self.screen.get_width()//2, self.screen.get_height()//2), pycircle.height//2, 5)

    # def handle_notes(self):
    #     if len(self.notes) == 0:
    #         prev_note_x = NOTE_START_X
    #         prev_note_y = random.randint(
    #             math.ceil(BAR_Y + BAR_HEIGHT/2), math.ceil(BAR_Y + BAR_HEIGHT - NOTE_HEIGHT))

    #     if len(self.notes) < RhythmGame.min_notes:
    #         note_y = random.randint(math.ceil(BAR_Y), math.ceil(BAR_Y +
    #                                                             BAR_HEIGHT - NOTE_HEIGHT))
    #         if len(self.notes) > 0:
    #             prev_note_x = self.notes[-1].get_x()
    #             prev_note_y = self.notes[-1].get_y()
    #             RhythmGame.notes_distance = RhythmGame.unit_time * \
    #                 random.randint(1, 4)
    #             note_x = prev_note_x + RhythmGame.notes_distance
    #         else:
    #             note_x = NOTE_START_X

    #         if self.is_valid_note(note_x, note_y, prev_note_x, prev_note_y):
    #             self.create_note(note_x, note_y)
    #         else:
    #             rv = random.randint(1, 4)  # 1, 2, 3, 4
    #             if (rv == 1):
    #                 # increase x 25%
    #                 note_x += RhythmGame.notes_distance * random.randint(1, 2)
    #             else:
    #                 # or decrease y 75%
    #                 offset = prev_note_y - note_y
    #                 note_y += offset*0.25*random.randint(2, 4)
    #             self.create_note(note_x, note_y)

    def is_valid_note(self, x, y, prev_x, prev_y) -> bool:
        if abs(x - prev_x) <= 750 and abs(y - prev_y) >= 100:
            return False
        return True


def main():
    game = RhythmGame()

    # create a random number generator
    random.seed(time.time())

    # recorder, trigger, receiver = testRecorded.init_stream(
    #     bufsize=0.1, winsize=0.1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # testRecorded.close_stream(trigger, recorder)
                sys.exit()

    #     filtered_data = testRecorded.filter_data(
    #         receiver=receiver, seconds_sleep=0, padlen=2)
    #     average_voltages = testRecorded.average_voltages(filtered_data)
    #     # print("ave_volt:", average_voltages.mean())
        # game.cursor.eeg_move(average_voltages.mean())
        # game.cursor.move()
        game.update_screen()
        game.display()


if __name__ == "__main__":
    main()
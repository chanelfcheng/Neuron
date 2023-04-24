import pygame
import sys
import random
import time
import math
# import testRecorded

##### CONFIG ###########
pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = pygame.display.get_desktop_sizes()[0]

RADIUS = 70
OFFSET = 35

RED_HI_COLOR = pygame.Color(255, 0, 162)  # green
RED_LO_COLOR = pygame.Color(255//2, 0, 162//2)  # green

GREEN_HI_COLOR = pygame.Color(0, 255, 0)  # green
GREEN_LO_COLOR = pygame.Color(0, 255//2, 0)  # green

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

    def __init__(self, bar: pygame.Rect, color) -> None:
        self.bar = bar
        self.color = color
        self.cursor_touch = False
        self.frame_counter = 1

    def move(self, step_size=1):
        self.bar = self.bar.move(step_size, 0)
        
    def get_color(self):
        return self.color
    
    def get_radius(self):
        return self.bar.height//2

    def get_bar(self) -> pygame.Rect:
        return self.bar

    def get_x(self) -> int:
        return self.bar.left

    def get_y(self) -> int:
        return self.bar.top

    # unused anymore
    # def is_touching_center(self, cursor_x, cursor_y):
    #     if cursor_x <= self.get_x() and self.get_x() <= (cursor_x + CURSOR_WIDTH):
    #         if self.get_y() >= cursor_y and self.get_y() <= cursor_y + CURSOR_HEIGHT:
    #             return True
    #     return False


class RhythmGame:
    
    # Display
    window_width = 0.9 * WINDOW_WIDTH
    window_height = 0.9 * WINDOW_HEIGHT
    screen = pygame.display.set_mode(
            (window_width, window_height), pygame.SCALED)
    
    # Notes
    unit_time = 250
    notes_distance = 1000
    num_notes = [4, 8, 12]
    max_notes = 16

    green_circle_x = screen.get_width()//4
    red_circle_x = 3*screen.get_width()//4
    circle_y = screen.get_height()//2

    def __init__(self) -> None:
        # self.cursor = Cursor()
        self.red_notes = []
        self.green_notes = []
        self.hi_flag = False
        self.start_time = time.time()
        self.user_turn = False
        self.num_notes = RhythmGame.num_notes[0]
        self.red_note_counter = 0
        self.green_note_counter = 0
        self.is_pressing_up = False
        self.is_pressing_down = False

        # TODO: draw menu screen

    def add_red_hi_note(self):

        bar = pygame.draw.rect(self.screen, RED_HI_COLOR, 
                               pygame.Rect(RhythmGame.red_circle_x - 2*RADIUS, 
                                RhythmGame.circle_y - OFFSET, 70, 10))
        
        note = Note(bar, RED_HI_COLOR)
        self.red_notes.append(note)

    def add_red_lo_note(self):
        
        bar = pygame.draw.rect(self.screen, RED_LO_COLOR, 
                               pygame.Rect(RhythmGame.red_circle_x - 2*RADIUS, 
                                RhythmGame.circle_y + OFFSET, 70, 10))
        
        note = Note(bar, RED_LO_COLOR)
        self.red_notes.append(note)

    def add_green_hi_note(self):

        bar = pygame.draw.rect(self.screen, GREEN_HI_COLOR, 
                               pygame.Rect(RhythmGame.green_circle_x + RADIUS, 
                                RhythmGame.circle_y - OFFSET, 70, 10))
        
        note = Note(bar, GREEN_HI_COLOR)
        self.green_notes.append(note)

    def add_green_lo_note(self):
        
        bar = pygame.draw.rect(self.screen, GREEN_LO_COLOR, 
                               pygame.Rect(RhythmGame.green_circle_x + RADIUS, 
                                RhythmGame.circle_y + OFFSET, 70, 10))
        
        note = Note(bar, GREEN_LO_COLOR)
        self.green_notes.append(note)

    def update_screen(self) -> None:
        self.screen.fill((0, 0, 0))  # black
        self.draw_circles()

        if (time.time() - self.start_time > 1):
            for note in self.red_notes:
                note.move(-140)
            
            for note in self.green_notes:
                note.move(140)

            self.start_time = time.time()
            if not self.user_turn:
                self.add_red_hi_note() if self.hi_flag else self.add_red_lo_note()
                self.hi_flag = not self.hi_flag
                self.red_note_counter += 1
                if self.red_note_counter == self.num_notes:
                    self.user_turn = True
                    self.red_note_counter = 0

            if self.user_turn:
                key = pygame.key.get_pressed()

                # up_pressed = key[pygame.K_UP]
                if key[pygame.K_UP] and not self.is_pressing_up:
                    # self.is_pressing_up = True
                    self.add_green_hi_note()
                    self.green_note_counter += 1
                    print("generating green hi note")
                # self.key_pressed = up_pressed

                down_pressed = key[pygame.K_DOWN]
                if key[pygame.K_DOWN] and not self.is_pressing_down:
                    # self.is_pressing_down = True
                    self.add_green_lo_note()
                    self.green_note_counter += 1
                    print("generating green lo note")
                    
                # self.key_pressed = down_pressed
                # if not up_pressed and self.is_pressing_up:
                    # self.is_pressing_up = False
                # if not down_pressed and self.is_pressing_down:
                    # self.is_pressing_down = False
                
                if self.green_note_counter == self.num_notes:
                    self.user_turn = False
                    self.green_note_counter = 0

        # draw/remove notes  
        for note in self.red_notes:
            # get the note's obj(rect)
            self.draw_note(note.get_bar(), note.get_color())
            if note.get_x() < self.green_circle_x + RADIUS:
                self.red_notes.remove(note)
 
        for note in self.green_notes:
            # get the note's obj(rect)
            self.draw_note(note.get_bar(), note.get_color())
            if note.get_x() > self.red_circle_x - RADIUS - note.get_bar().width:
                self.green_notes.remove(note)       

            

    def display(self) -> None:
        pygame.display.flip()

    def draw_circles(self) -> None:
        # GREEN CIRCLE
        note_color = pygame.Color(0, 255-100, 162-100)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//4, self.screen.get_height()//2), RADIUS, 0)
        
        note_color = pygame.Color(0, 255-50, 162-50)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//4, self.screen.get_height()//2), 60, 0)
        
        note_color = pygame.Color(0, 255, 162)
        pygame.draw.circle(self.screen, note_color,
                                (self.screen.get_width()//4, self.screen.get_height()//2), 50, 0)

        # RED CIRCLE
        note_color = pygame.Color(255-100, 0, 100-100)
        pygame.draw.circle(self.screen, note_color,
                                (3*self.screen.get_width()//4, self.screen.get_height()//2), RADIUS, 0)
        
        note_color = pygame.Color(255-50, 0, 100-50)
        pygame.draw.circle(self.screen, note_color,
                                (3*self.screen.get_width()//4, self.screen.get_height()//2), 60, 0)
        
        note_color = pygame.Color(255, 0, 100)
        pygame.draw.circle(self.screen, note_color,
                                (3*self.screen.get_width()//4, self.screen.get_height()//2), 50, 0)

    def draw_note(self, bar, color) -> None:
        # pygame.draw.circle(self.screen, NOTE_COLOR,
        #                             (self.screen.get_width()//2, self.screen.get_height()//2), pycircle.height//2, 5)
        pygame.draw.rect(self.screen, color, bar)

    # def handle_notes(self):
    #     if len(self.red_notes) == 0:
    #         prev_note_x = NOTE_START_X
    #         prev_note_y = random.randint(
    #             math.ceil(BAR_Y + BAR_HEIGHT/2), math.ceil(BAR_Y + BAR_HEIGHT - NOTE_HEIGHT))

    #     if len(self.red_notes) < RhythmGame.min_notes:
    #         note_y = random.randint(math.ceil(BAR_Y), math.ceil(BAR_Y +
    #                                                             BAR_HEIGHT - NOTE_HEIGHT))
    #         if len(self.red_notes) > 0:
    #             prev_note_x = self.red_notes[-1].get_x()
    #             prev_note_y = self.red_notes[-1].get_y()
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

    game.display()
    pygame.event.get()

    # create a random number generator
    random.seed(time.time())

    time.sleep(1)

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
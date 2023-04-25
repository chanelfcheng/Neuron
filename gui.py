import pygame
import sys
import random
import time
from itertools import repeat
# import testRecorded

# GAME CONSTANTS
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = pygame.display.get_desktop_sizes()[0]

CIRCLE_RADIUS = 70
HI_LO_HEIGHT = 35
NOTE_WIDTH = 118
NOTE_HEIGHT = 4

RED_HI_COLOR = pygame.Color(200, 0, 100)  # green
RED_LO_COLOR = pygame.Color(200//2, 0, 100//2)  # green

GREEN_HI_COLOR = pygame.Color(0, 200, 100)  # green
GREEN_LO_COLOR = pygame.Color(0, 200//2, 100//2)  # green

HI_AUDIO = "audio/high.mp3"
LO_AUDIO = "audio/low.mp3"

def shake():
    s = -1
    for _ in range(0, 3):
        for x in range(0, 20, 5):
            yield (x*s, 0)
        for x in range(20, 0, 5):
            yield (x*s, 0)
        s *= -1
    while True:
        yield (0, 0)

class Note:

    def __init__(self, bar: pygame.Rect, color) -> None:
        self.bar = bar
        self.color = color
        self.cursor_touch = False
        self.frame_counter = 1

    def move(self, step_size=1):
        self.bar = self.bar.move(step_size, 0)
    
    def get_type(self):
        if self.color == RED_HI_COLOR or self.color == GREEN_HI_COLOR:
            return 1
        else:
            return 0

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
    
    def __eq__(self, other) -> bool:
        return self.get_type() == other.get_type()


class RhythmGame:
    
    # Display
    window_width = 0.75 * WINDOW_WIDTH
    window_height = 0.75 * WINDOW_HEIGHT
    screen = pygame.display.set_mode(
            (window_width, window_height), pygame.SCALED)
    
    pygame.display.set_caption("Neuron")

    green_circle_x = 4*CIRCLE_RADIUS
    red_circle_x = screen.get_width() - 4*CIRCLE_RADIUS
    circle_y = screen.get_height()//2

    num_notes = [4, 8, 16, 32]

    def __init__(self) -> None:
        # self.cursor = Cursor()
        self.red_notes = []
        self.green_notes = []
        self.input_notes = []
        self.correct = None
        self.gen_note_type = 0
        self.prev_note_type = 0
        
        self.start_time = time.time()
        self.user_turn = False

        self.level = 0
        self.time_gap = 1.0
        self.red_note_counter = 0
        self.green_note_counter = 0
        self.correct_note_counter = 0

        self.is_pressing_up = False
        self.is_pressing_down = False
        self.menu_screen = True

        self.offset = repeat((0,0))
        pygame.mixer.init()

        # draw menu screen
        self.screen.fill((255, 255, 255))
        self.draw_circles()
        text = pygame.freetype.Font("fonts/pixel.ttf", 150)
        text.render_to(RhythmGame.screen, 
                        (RhythmGame.window_width//6, RhythmGame.window_height//2), 
                        "START GAME", 
                        (0, 0, 0))
        

    def add_red_hi_note(self):

        bar = pygame.draw.rect(self.screen, RED_HI_COLOR, 
                               pygame.Rect(RhythmGame.red_circle_x - CIRCLE_RADIUS - NOTE_WIDTH, 
                                RhythmGame.circle_y - HI_LO_HEIGHT, NOTE_WIDTH, NOTE_HEIGHT))
        
        note = Note(bar, RED_HI_COLOR)
        self.red_notes.append(note)

    def add_red_lo_note(self):
        
        bar = pygame.draw.rect(self.screen, RED_LO_COLOR, 
                               pygame.Rect(RhythmGame.red_circle_x - CIRCLE_RADIUS - NOTE_WIDTH, 
                                RhythmGame.circle_y + HI_LO_HEIGHT, NOTE_WIDTH, NOTE_HEIGHT))
        
        note = Note(bar, RED_LO_COLOR)
        self.red_notes.append(note)

    def add_green_hi_note(self):

        bar = pygame.draw.rect(self.screen, GREEN_HI_COLOR, 
                               pygame.Rect(RhythmGame.green_circle_x + CIRCLE_RADIUS, 
                                RhythmGame.circle_y - HI_LO_HEIGHT, NOTE_WIDTH, NOTE_HEIGHT))
        
        note = Note(bar, GREEN_HI_COLOR)
        self.green_notes.append(note)

    def add_green_lo_note(self):
        
        bar = pygame.draw.rect(self.screen, GREEN_LO_COLOR, 
                               pygame.Rect(RhythmGame.green_circle_x + CIRCLE_RADIUS, 
                                RhythmGame.circle_y + HI_LO_HEIGHT, NOTE_WIDTH, NOTE_HEIGHT))
        
        note = Note(bar, GREEN_LO_COLOR)
        self.green_notes.append(note)

    def move_notes(self):
        for note in self.red_notes:
            note.move(-NOTE_WIDTH)
        
        for note in self.green_notes:
            note.move(NOTE_WIDTH)

    def add_notes(self):
        self.start_time = time.time()
        if not self.user_turn:
            if self.red_note_counter != RhythmGame.num_notes[self.level]:
                if self.gen_note_type == 1:
                    self.prev_note_type = 1
                    self.add_red_hi_note()
                    pygame.mixer.music.load(HI_AUDIO)
                    pygame.mixer.music.play()
                elif self.gen_note_type == 0:
                    self.prev_note_type = 0
                    self.add_red_lo_note()
                    pygame.mixer.music.load(LO_AUDIO)
                    pygame.mixer.music.play()
                self.gen_note_type = random.choice([0, 1])
                self.red_note_counter += 1
            else:
                if len(self.red_notes) == 0:
                    self.user_turn = True
                    self.red_note_counter = 0

        if self.user_turn:
            if self.green_note_counter != RhythmGame.num_notes[self.level]:
                key = pygame.key.get_pressed()

                # up_pressed = key[pygame.K_UP]
                if key[pygame.K_UP]: # and not self.is_pressing_up:
                    # self.is_pressing_up = True
                    self.add_green_hi_note()
                    pygame.mixer.music.load(HI_AUDIO)
                    pygame.mixer.music.play()
                    self.green_note_counter += 1
                # self.key_pressed = up_pressed

                # down_pressed = key[pygame.K_DOWN]
                elif key[pygame.K_DOWN]: # and not self.is_pressing_down:
                    # self.is_pressing_down = True
                    self.add_green_lo_note()
                    pygame.mixer.music.load(LO_AUDIO)
                    pygame.mixer.music.play()
                    self.green_note_counter += 1
                    
                # self.key_pressed = down_pressed
                # if not up_pressed and self.is_pressing_up:
                    # self.is_pressing_up = False
                # if not down_pressed and self.is_pressing_down:
                    # self.is_pressing_down = False
            else:
                if len(self.green_notes) == 0:
                    self.correct = None
                    if self.correct_note_counter / RhythmGame.num_notes[self.level] >= 0.7:
                        text = pygame.freetype.Font("fonts/pixel.ttf", 50)
                        text.render_to(RhythmGame.screen, 
                                       (RhythmGame.window_width//2, RhythmGame.window_height//2), 
                                       "NEXT WAVE?", 
                                       (0, 0, 0))
                        time.sleep(1)
                    else:
                        text = pygame.freetype.Font("fonts/pixel.ttf", 50)
                        text.render_to(RhythmGame.screen, 
                                       (RhythmGame.window_width//2, RhythmGame.window_height//2), 
                                       "TRY AGAIN?", 
                                       (0, 0, 0))
                        time.sleep(1)
                    key = pygame.key.get_pressed()
                
                    if key[pygame.K_UP]:
                        if self.correct_note_counter / RhythmGame.num_notes[self.level] >= 0.7:
                            if self.time_gap > 0.8:
                                self.time_gap -= 0.1
                            
                            else:
                                if self.level <= 2:
                                    print("Next level")
                                    self.level += 1
                                    self.time_gap = 1.0
                            # self.level += 1 if self.level <= 2 and self.time_gap <= 0.5 else 0
                        self.correct_note_counter = 0
                        self.user_turn = False
                        self.green_note_counter = 0

    def draw_dashed_line(self, color):
        for i in range(0, int(RhythmGame.window_width), 10):
            pygame.draw.rect(self.screen, color, 
                             pygame.Rect(i, RhythmGame.window_height//2 - 2, 4, 1))

    def render_notes(self):
        prev_note = None
        for note in self.red_notes:
            if note.get_x() < self.green_circle_x + CIRCLE_RADIUS:
                self.red_notes.remove(note)
                self.input_notes.append(note)
            else:
                self.draw_note(note.get_bar(), note.get_color())
                if prev_note is not None and prev_note != note:
                    self.draw_line(note.get_bar().x, note.get_color())
            prev_note = note
 
        prev_note = None
        for note in self.green_notes:
            input_note = self.input_notes[0]
            if note.get_x() > self.red_circle_x - CIRCLE_RADIUS - note.get_bar().width:
                self.green_notes.remove(note)
                self.input_notes.remove(input_note)
                self.check_correct(input_note, note)
            else:
                self.draw_note(note.get_bar(), note.get_color())
                if prev_note is not None and prev_note != note:
                    self.draw_line(note.get_bar().x + note.get_bar().width, note.get_color())
            prev_note = note

    def check_correct(self, input_note, note):
        if note == input_note:
            self.correct_note_counter += 1
            self.correct = True
        else:
            self.offset = shake()
            self.correct = False

    def display_performance(self):
        # display percentage of correct notes
        text = pygame.freetype.Font("fonts/pixel.ttf", 50)
        text.render_to(RhythmGame.screen, 
                       (RhythmGame.red_circle_x, RhythmGame.window_height//2), 
                       f"{self.correct_note_counter / RhythmGame.num_notes[self.level] * 100}%", 
                       (0, 0, 0))


    def update_screen(self) -> None:
        self.screen.fill((255, 255, 255))  # black

        if self.user_turn:
            self.draw_dashed_line(pygame.Color(0, 200, 100))
        else:
            self.draw_dashed_line(pygame.Color(200, 0, 100))

        self.draw_circles()
        self.display_performance()

        if time.time() - self.start_time > self.time_gap:
            self.move_notes()
            self.add_notes()

        self.render_notes()

        if self.correct == True:
            text = pygame.freetype.Font("fonts/pixel.ttf", 50)
            text.render_to(RhythmGame.screen, 
                           (RhythmGame.red_circle_x - 2*CIRCLE_RADIUS, RhythmGame.window_height//2), 
                           ":)", (0, 0, 0))
        elif self.correct == False:
            text = pygame.freetype.Font("fonts/pixel.ttf", 50)
            text.render_to(RhythmGame.screen, 
                           (RhythmGame.red_circle_x - 2*CIRCLE_RADIUS, RhythmGame.window_height//2), 
                           ":(", (0, 0, 0))  

    def display(self) -> None:
        self.screen.blit(self.screen, next(self.offset))
        pygame.display.flip()

    def draw_circles(self) -> None:
        # GREEN CIRCLE
        note_color = pygame.Color(0, 255-100, 162-100)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.green_circle_x, RhythmGame.circle_y), CIRCLE_RADIUS, 0)
        
        note_color = pygame.Color(0, 255-50, 162-50)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.green_circle_x, RhythmGame.circle_y), 60, 0)
        
        note_color = pygame.Color(0, 255, 162)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.green_circle_x, RhythmGame.circle_y), 50, 0)

        # RED CIRCLE
        note_color = pygame.Color(255-100, 0, 100-100)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.red_circle_x, RhythmGame.circle_y), CIRCLE_RADIUS, 0)
        
        note_color = pygame.Color(255-50, 0, 100-50)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.red_circle_x, RhythmGame.circle_y), 60, 0)
        
        note_color = pygame.Color(255, 0, 100)
        pygame.draw.circle(self.screen, note_color,
                                (RhythmGame.red_circle_x, RhythmGame.circle_y), 50, 0)

    def draw_note(self, bar, color) -> None:
        pygame.draw.rect(self.screen, color, bar)
        
    def draw_line(self, x, color):
        pygame.draw.rect(self.screen, color,
                        pygame.Rect(x, 
                        RhythmGame.window_height//2 - HI_LO_HEIGHT, 
                        4, 2*HI_LO_HEIGHT+4))

def main():
    game = RhythmGame()

    game.display()
    pygame.event.get()

    time.sleep(1)
    while game.menu_screen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # testRecorded.close_stream(trigger, recorder)
                sys.exit()
            elif event.type == pygame.KEYUP:
                game.menu_screen = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        game.update_screen()
        game.display()


if __name__ == "__main__":
    main()
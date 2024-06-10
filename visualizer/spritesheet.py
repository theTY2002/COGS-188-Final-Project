# Adapted from https://ehmatthes.github.io/pcc_2e/beyond_pcc/pygame_sprite_sheets/
# Which is adapted from https://www.pygame.org/wiki/Spritesheet
# Which is adapted from https://www.scriptefun.com/transcript-2-using-sprite-sheets-and-drawing-the-background

import pygame as pg

class SpriteSheet:
    # The full sheet
    sheet: pg.Surface

    # The size of each item
    size: pg.Rect

    # The margin around each item (left, top, right, bottom)
    margin: tuple[int, int, int, int]

    def __init__(self, filename: str, size: pg.Rect, margin: tuple[int, int] | tuple[int, int, int, int] = (0, 0)):
        """Load the sheet."""
        try:
            self.sheet = pg.image.load(filename).convert_alpha()
            self.size = size
            self.margin = margin if len(margin) == 4 else (margin[0], margin[1], margin[0], margin[1])
        except pg.error as e:
            print(f"Unable to load spritesheet image: {filename}")
            raise SystemExit(e)

    def image_at(self, x: int, y: int) -> tuple[pg.Surface, pg.Rect]:
        """Load a specific image from a specific rectangle."""
        rect = pg.Rect(
            x * self.size.width + self.margin[0], 
            y * self.size.height + self.margin[1], 
            self.size.width - self.margin[2], 
            self.size.height - self.margin[3]
        )
        image = self.sheet.subsurface(rect)
        return image, image.get_rect()

from pygame.sprite import Sprite, Group

from simulator.tiles import Tile, SuitedTile, WindTile, DragonTile, SeasonTile, FlowerTile
from visualizer.spritesheet import SpriteSheet

class TileSprite(Sprite):
    tile: Tile
    sheet: SpriteSheet

    def __init__(self, tile: Tile, sheet: SpriteSheet):
        super().__init__()
        self.tile = tile

        match tile:
            case SuitedTile(suit, rank):
                # Load the image and rect from this position (first 3 rows are suited tiles)
                self.image, self.rect = sheet.image_at(rank - 1, suit.value - 1)
            case WindTile(wind):
                # Load the image and rect from the first half of the 4th row (wind tiles)
                self.image, self.rect = sheet.image_at(wind.value - 1, 3)
            case DragonTile() as dragon:
                # Load the image and rect from the second half of the 4th row (dragon tiles)
                self.image, self.rect = sheet.image_at(4 + dragon.value - 1, 3)
            case SeasonTile(wind):
                # Load the image and rect from the first half of the 5th row (season tiles)
                self.image, self.rect = sheet.image_at(wind.value - 1, 4)
            case FlowerTile(wind):
                # Load the image and rect from the second half of 5th row (flower tiles)
                self.image, self.rect = sheet.image_at(4 + wind.value - 1, 4)

class TileGroup(Group):
    def __init__(self, hand: list[Tile], sheet: SpriteSheet):
        super().__init__()
        for tile in hand:
            self.add(TileSprite(tile, sheet))
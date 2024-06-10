import pygame as pg

from simulator.tiles import Suit, SuitedTile
from visualizer.components import TileSprite
from visualizer.spritesheet import SpriteSheet


if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode((800, 800), pg.SCALED)
    pg.display.set_caption("Mahjong")
    pg.mouse.set_visible(True)

    background = pg.Surface(screen.get_size()).convert()
    background.fill((50,) * 3)

    tile_sheet = SpriteSheet("assets/deck_mahjong_light_1.png", pg.Rect(0, 0, 64, 64), (10, 2))

    tile_sprite = TileSprite(SuitedTile(Suit.BAMBOO, 1), tile_sheet)

    sprites = pg.sprite.Group((tile_sprite))

    clock = pg.time.Clock()

    going = True
    while going:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                going = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                pass
            elif event.type == pg.MOUSEBUTTONUP:
                pass

        sprites.update()

        tile_sprite.rect.center = pg.mouse.get_pos()

        screen.blit(background, (0, 0))
        sprites.draw(screen)
        pg.display.flip()

        clock.tick(60)

    pg.quit()

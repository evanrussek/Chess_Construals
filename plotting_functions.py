from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import chess
from chess.svg import *
from chess.svg import _svg, _select_color, _attrs, _coord, _color, Color, IntoSquareSet, Square

from typing import Dict, Iterable, Optional, Tuple, Union

import pandas as pd
import numpy as np

def plot_heat_map(piece_heat_map, board: Optional[chess.BaseBoard] = None, *,
          orientation: Color = chess.WHITE,
          lastmove: Optional[chess.Move] = None,
          check: Optional[Square] = None,
          arrows: Iterable[Union[Arrow, Tuple[Square, Square]]] = [],
          fill: Dict[Square, str] = {},
          squares: Optional[IntoSquareSet] = None,
          size: Optional[int] = None,
          coordinates: bool = True,
          colors: Dict[str, str] = {},
          flipped: bool = False,
          style: Optional[str] = None) -> str:
    """
    Renders a board with pieces and/or selected squares as an SVG image.

    :param board: A :class:`chess.BaseBoard` for a chessboard with pieces, or
        ``None`` (the default) for a chessboard without pieces.
    :param orientation: The point of view, defaulting to ``chess.WHITE``.
    :param lastmove: A :class:`chess.Move` to be highlighted.
    :param check: A square to be marked indicating a check.
    :param arrows: A list of :class:`~chess.svg.Arrow` objects, like
        ``[chess.svg.Arrow(chess.E2, chess.E4)]``, or a list of tuples, like
        ``[(chess.E2, chess.E4)]``. An arrow from a square pointing to the same
        square is drawn as a circle, like ``[(chess.E2, chess.E2)]``.
    :param fill: A dictionary mapping squares to a colors that they should be
        filled with.
    :param squares: A :class:`chess.SquareSet` with selected squares to mark
        with an X.
    :param size: The size of the image in pixels (e.g., ``400`` for a 400 by
        400 board), or ``None`` (the default) for no size limit.
    :param coordinates: Pass ``False`` to disable the coordinate margin.
    :param colors: A dictionary to override default colors. Possible keys are
        ``square light``, ``square dark``, ``square light lastmove``,
        ``square dark lastmove``, ``margin``, ``coord``, ``arrow green``,
        ``arrow blue``, ``arrow red``, and ``arrow yellow``. Values should look
        like ``#ffce9e`` (opaque), or ``#15781B80`` (transparent).
    :param flipped: Pass ``True`` to flip the board.
    :param style: A CSS stylesheet to include in the SVG image.

    >>> import chess
    >>> import chess.svg
    >>>
    >>> board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
    >>>
    >>> chess.svg.board(
    ...     board,
    ...     fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc") | {chess.E4: "#00cc00cc"},
    ...     arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
    ...     squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
    ...     size=350,
    ... )  # doctest: +SKIP

    .. image:: ../docs/Ne4.svg
        :alt: 8/8/8/8/4N3/8/8/8

    .. deprecated:: 1.1
        Use *orientation* with a color instead of the *flipped* toggle.
    """
    orientation ^= flipped
    margin = 15 if coordinates else 0
    svg = _svg(8 * SQUARE_SIZE + 2 * margin, size)

    if style:
        ET.SubElement(svg, "style").text = style

    if board:
        desc = ET.SubElement(svg, "desc")
        asciiboard = ET.SubElement(desc, "pre")
        asciiboard.text = str(board)

    defs = ET.SubElement(svg, "defs")
    if board:
        for piece_color in chess.COLORS:
            for piece_type in chess.PIECE_TYPES:
                if board.pieces_mask(piece_type, piece_color):
                    defs.append(ET.fromstring(PIECES[chess.Piece(piece_type, piece_color).symbol()]))

    squares = chess.SquareSet(squares) if squares else chess.SquareSet()
    if squares:
        defs.append(ET.fromstring(XX))        
        
    squares = chess.SquareSet(squares) if squares else chess.SquareSet()
    if squares:
        defs.append(ET.fromstring(XX))

    if check is not None:
        defs.append(ET.fromstring(CHECK_GRADIENT))

    # Render coordinates.
    if coordinates:
        margin_color, margin_opacity = _select_color(colors, "margin")
        ET.SubElement(svg, "rect", _attrs({
            "x": 0,
            "y": 0,
            "width": 2 * margin + 8 * SQUARE_SIZE,
            "height": 2 * margin + 8 * SQUARE_SIZE,
            "fill": margin_color,
            "opacity": margin_opacity if margin_opacity < 1.0 else None,
        }))
        coord_color, coord_opacity = _select_color(colors, "coord")
        for file_index, file_name in enumerate(chess.FILE_NAMES):
            x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + margin
            svg.append(_coord(file_name, x, 0, SQUARE_SIZE, margin, True, margin, color=coord_color, opacity=coord_opacity))
            svg.append(_coord(file_name, x, margin + 8 * SQUARE_SIZE, SQUARE_SIZE, margin, True, margin, color=coord_color, opacity=coord_opacity))
        for rank_index, rank_name in enumerate(chess.RANK_NAMES):
            y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + margin
            svg.append(_coord(rank_name, 0, y, margin, SQUARE_SIZE, False, margin, color=coord_color, opacity=coord_opacity))
            svg.append(_coord(rank_name, margin + 8 * SQUARE_SIZE, y, margin, SQUARE_SIZE, False, margin, color=coord_color, opacity=coord_opacity))

    # Render board.
    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + margin
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + margin

        cls = ["square", "light" if chess.BB_LIGHT_SQUARES & bb else "dark"]
        if lastmove and square in [lastmove.from_square, lastmove.to_square]:
            cls.append("lastmove")
        square_color, square_opacity = _select_color(colors, " ".join(cls))

        cls.append(chess.SQUARE_NAMES[square])

        ET.SubElement(svg, "rect", _attrs({
            "x": x,
            "y": y,
            "width": SQUARE_SIZE,
            "height": SQUARE_SIZE,
            "class": " ".join(cls),
            "stroke": "none",
            "fill": square_color,
            "opacity": square_opacity if square_opacity < 1.0 else None,
        }))

        try:
            fill_color, fill_opacity = _color(fill[square])
        except KeyError:
            pass
        else:
            ET.SubElement(svg, "rect", _attrs({
                "x": x,
                "y": y,
                "width": SQUARE_SIZE,
                "height": SQUARE_SIZE,
                "stroke": "none",
                "fill": fill_color,
                "opacity": fill_opacity if fill_opacity < 1.0 else None,
            }))

    # Render check mark.
    if check is not None:
        file_index = chess.square_file(check)
        rank_index = chess.square_rank(check)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + margin
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + margin

        ET.SubElement(svg, "rect", _attrs({
            "x": x,
            "y": y,
            "width": SQUARE_SIZE,
            "height": SQUARE_SIZE,
            "class": "check",
            "fill": "url(#check_gradient)",
        }))

    # Render pieces and selected squares.
    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + margin
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + margin

        if piece_heat_map is not None:
            for piece_type in chess.PIECE_TYPES:
                for piece_color in [chess.WHITE,chess.BLACK]:
                    piece_opacity = piece_heat_map[square,piece_type+(piece_color==chess.WHITE)*7]
                    if piece_opacity>0:
                        href = f"#{chess.COLOR_NAMES[piece_color]}-{chess.PIECE_NAMES[piece_type]}"
                        #print(href,x,y,piece_opacity)
                        ET.SubElement(svg, "use", _attrs({
                            "href": href,
                            "xlink:href": href,
                            "transform": f"translate({x:d}, {y:d})",
                            "opacity": piece_opacity if piece_opacity < 1.0 else None,
                        }))
                        
        # Render selected squares.
        if squares is not None and square in squares:
            ET.SubElement(svg, "use", _attrs({
                "href": "#xx",
                "xlink:href": "#xx",
                "x": x,
                "y": y,
            }))

    # Render arrows.
    for arrow in arrows:
        try:
            tail, head, color = arrow.tail, arrow.head, arrow.color  # type: ignore
        except AttributeError:
            tail, head = arrow  # type: ignore
            color = "green"

        try:
            color, opacity = _select_color(colors, " ".join(["arrow", color]))
        except KeyError:
            opacity = 1.0

        tail_file = chess.square_file(tail)
        tail_rank = chess.square_rank(tail)
        head_file = chess.square_file(head)
        head_rank = chess.square_rank(head)

        xtail = margin + (tail_file + 0.5 if orientation else 7.5 - tail_file) * SQUARE_SIZE
        ytail = margin + (7.5 - tail_rank if orientation else tail_rank + 0.5) * SQUARE_SIZE
        xhead = margin + (head_file + 0.5 if orientation else 7.5 - head_file) * SQUARE_SIZE
        yhead = margin + (7.5 - head_rank if orientation else head_rank + 0.5) * SQUARE_SIZE

        if (head_file, head_rank) == (tail_file, tail_rank):
            ET.SubElement(svg, "circle", _attrs({
                "cx": xhead,
                "cy": yhead,
                "r": SQUARE_SIZE * 0.9 / 2,
                "stroke-width": SQUARE_SIZE * 0.1,
                "stroke": color,
                "opacity": opacity if opacity < 1.0 else None,
                "fill": "none",
                "class": "circle",
            }))
        else:
            marker_size = 0.75 * SQUARE_SIZE
            marker_margin = 0.1 * SQUARE_SIZE

            dx, dy = xhead - xtail, yhead - ytail
            hypot = math.hypot(dx, dy)

            shaft_x = xhead - dx * (marker_size + marker_margin) / hypot
            shaft_y = yhead - dy * (marker_size + marker_margin) / hypot

            xtip = xhead - dx * marker_margin / hypot
            ytip = yhead - dy * marker_margin / hypot

            ET.SubElement(svg, "line", _attrs({
                "x1": xtail,
                "y1": ytail,
                "x2": shaft_x,
                "y2": shaft_y,
                "stroke": color,
                "opacity": opacity if opacity < 1.0 else None,
                "stroke-width": SQUARE_SIZE * 0.2,
                "stroke-linecap": "butt",
                "class": "arrow",
            }))

            marker = [(xtip, ytip),
                      (shaft_x + dy * 0.5 * marker_size / hypot,
                       shaft_y - dx * 0.5 * marker_size / hypot),
                      (shaft_x - dy * 0.5 * marker_size / hypot,
                       shaft_y + dx * 0.5 * marker_size / hypot)]

            ET.SubElement(svg, "polygon", _attrs({
                "points": " ".join(f"{x},{y}" for x, y in marker),
                "fill": color,
                "opacity": opacity if opacity < 1.0 else None,
                "class": "arrow",
            }))                        

    return SvgWrapper(ET.tostring(svg).decode("utf-8"))

def plot_samples(original_fen,selected_move,board_sample_df):
    original_board = chess.Board(original_fen)
    df = board_sample_df.copy()
    df['lik']/=df['lik'].sum()

    piece_heat_map = np.zeros([64,14])
    for _,fen,lik in df.itertuples():
        for square,piece in chess.Board(fen).piece_map().items():
            piece_heat_map[square,piece.piece_type+(piece.color==chess.WHITE)*7]+=lik    
    
    return plot_heat_map(piece_heat_map,original_board,arrows=[chess.svg.Arrow.from_pgn(selected_move)])            
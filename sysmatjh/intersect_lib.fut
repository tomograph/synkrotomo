module Intersections = {
  type point  = ( f32, f32 )

  -- Doesn't futhark have tan already?
  let tan (a: f32): f32 = f32.sin(a) / f32.cos(a)

  -- Degrees to radians converter
  let deg2rad(deg: f32): f32 = (f32.pi / 180f32) * deg

  let distance ((x1, y1) : point) ((x2, y2) : point): f32 =
    f32.sqrt( (x2 - x1) ** 2.0f32 + (y2 - y1) ** 2.0f32 )

  -- let fleil (flag : f32) (num : f32) : f32 =
  --   if flag < 0f32 then f32.ceil num else f32.floor num


  -- Check whether a point is within the grid.
  let isInGrid (g : f32) (y_step_dir : f32) ((x, y) : point) : bool =
    x >= 0f32 && x < g && (
      if y_step_dir == -1f32
      then (0f32 < y && y <= g)
      else (0f32 <= y && y < g)
    )

  -- Calculate x from y and vice versa
  let calcXfromY (y : f32 ) (s : f32) (b : f32) : f32 = (y - b) / s
  let calcYfromX (x : f32 ) (s : f32) (b : f32) : f32 = s * x + b

  -- Calculate entry points
  let entryPoint (s : f32) (b : f32) (g : f32) : point =
    if 0f32 <= b && b <= g then (0f32, b)
    else
      let p_bottom = (calcXfromY g s b, g)
      let p_top    = (calcXfromY 0f32 s b, 0f32)
      in if p_bottom.1 < p_top.1 then p_bottom else p_top

  let lengths
    (grid_size: i32)
    (theta:     f32)
    (delta:     f32)
    (line_num:  i32): [](f32, i32) =

    let g = r32(grid_size)
    let i = r32(line_num)

    let horizontal = theta == 90f32
    let vertical   = theta == 0f32
    let slope    = tan (deg2rad(theta + 90f32))
    --distance to move in y direction in each step
    let y_offset = delta / f32.cos(deg2rad(90f32 - theta))

    let A = replicate (t32(g*2f32-1f32)) (-1f32, -1)
    -- center of grid
    let center = (g / 2f32, g / 2f32)

    let b = if horizontal then center.2 - 1f32 else center.2 - slope * center.1

    -- if a vertical line then the entry point of a ray is the point at y = top of grid x = center.1 + i * delta + delta/2
    -- always even number of rays
    let entry_point = if vertical then ((center.1 + i * delta + delta/2) , g) else entryPoint slope (i * y_offset + y_offset/2 + b) g

    let y_step_dir = if slope < 0f32 then -1f32 else 1f32
    let anchorX = f32.floor(entry_point.1) + 1f32
    let anchorY = if y_step_dir == -1f32
      then f32.ceil(entry_point.2) - 1f32
      else f32.floor(entry_point.2) + 1f32

    let (A, _, _, _, _) =
      loop (A, focusPoint, anchorX, anchorY, write_index) = (A, entry_point, anchorX, anchorY, 0)
      while ( isInGrid g y_step_dir focusPoint ) do
        let p_anchor_x = ( anchorX,  calcYfromX anchorX slope (b + y_offset * i))
        let p_anchor_y = if vertical then  (entry_point.1, anchorY) else ( calcXfromY anchorY slope (b + y_offset * i), anchorY)

        let dist_p_x = distance focusPoint p_anchor_x
        let dist_p_y = distance focusPoint p_anchor_y

        let y_floor = f32.floor(focusPoint.2)
        let y_comp =
          if (y_step_dir == -1f32 && focusPoint.2 - y_floor == 0f32)
          then focusPoint.2 - 1f32
          else y_floor

        let index = t32( f32.floor(focusPoint.1) + g * y_comp)

        in
          if horizontal then
            unsafe let A[write_index] = (dist_p_x, index)
            in (A, p_anchor_x, anchorX + 1f32, anchorY, write_index+1)
          else if vertical then
            unsafe let A[write_index] = (dist_p_y, index)
            in (A, p_anchor_y, anchorX, anchorY + y_step_dir, write_index+1)
          else
          if (f32.abs(dist_p_x - dist_p_y) > 0.000000001f32)
          then
            if ( dist_p_x < dist_p_y )
            then
              unsafe let A[write_index] = (dist_p_x, index)
              in (A, p_anchor_x, anchorX + 1f32, anchorY, write_index+1)
            else
              unsafe let A[write_index] = (dist_p_y, index)
              in (A, p_anchor_y, anchorX, anchorY + y_step_dir, write_index+1)
          else
              unsafe let A[write_index] = (dist_p_x, index)
              in (A, p_anchor_x, anchorX + 1f32, anchorY + y_step_dir, write_index+1)
    in A
    -- in [entry_point.1, entry_point.2, anchorX, anchorY ]
}

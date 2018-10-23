module Intersections = {
  type point  = ( f32, f32 )
  -- Degrees to radians converter
  let deg2rad(deg: f32): f32 = (f32.pi / 180f32) * deg

  -- find x given y
  let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
    if cost == 0 then ray else (ray-y*sint)/cost

  -- find y given x
  let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
    if sint == 0 then ray else (ray-x*cost)/sint

  -- entry point from sin/cos
  let entryPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : point =
    let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint)
    let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
    let p_top = (find_x maxval ray cost sint, maxval)
    in if f32.abs(p_left.2) <= maxval && sint != 0 then p_left else if p_bottom.1 <= p_top.1 then p_bottom else p_top

  -- convertion to sin/cos arrays of array of degrees
  let convert2sincos (angles: []f32) : []point =
    map (\t -> (f32.sin(deg2rad(t)),f32.cos(deg2rad(t)))) angles

  --convert to entry points !!!SHOULD ALSO return cost sint
  let convert2entry (angles: []f32) (rays: []f32) (maxval: f32): [] (point, point) =
    let sincos = convert2sincos angles
    let anglesrays = flatten(map (\t -> map(\r -> (t.1,t.2,r)) rays) sincos)
      -- zip sint cost rays where sint cost gets replicated and flattened to match size of rays
    --let anglesrays = zip3 (flatten (replicate sizerays sint)) (flatten (replicate sizerays cost)) (flatten (replicate sizeangles rays))
    in map(\(s,c,r) -> ((entryPoint s c r maxval), (s,c))) anglesrays

  let distance ((x1, y1) : point) ((x2, y2) : point): f32 =
    f32.sqrt( (x2 - x1) ** 2.0f32 + (y2 - y1) ** 2.0f32 )

  -- Check whether a point is within the grid.
  let isInGrid (halfsize : f32) (y_step_dir : f32) ((x, y) : point) : bool =
    x >= -1f32*halfsize && x < halfsize && (
      if y_step_dir == -1f32
      then (-1f32*halfsize < y && y <= halfsize)
      else (-1f32*halfsize <= y && y < halfsize)
    )

  let lengths
    (grid_size: i32)
    (sint: f32)
    (cost: f32)
    (entry_point: point): [](f32, i32) =

    let horizontal = cost == 0
    let vertical = f32.abs(cost) == 1

    let slope = cost/(-sint) -- tan(x+90) = -cot(x) = slope since the angles ar ethe normals of the line

    let size = r32(grid_size)
    let halfsize = size/2.0f32

    let A = replicate (t32(size*2f32-1f32)) (-1f32, -1)

    let y_step_dir = if slope < 0f32 then -1f32 else 1f32
    let anchorX = f32.floor(entry_point.1) + 1f32
    let anchorY = if y_step_dir == -1f32
      then f32.ceil(entry_point.2) - 1f32
      else f32.floor(entry_point.2) + 1f32

    let (A, _, _, _, _) =
      loop (A, focusPoint, anchorX, anchorY, write_index) = (A, entry_point, anchorX, anchorY, 0)
      while ( isInGrid halfsize y_step_dir focusPoint ) do
        --compute index of pixel in array by computing x component and y component if
        --center was at bottom left corner (add halfsize), and add them multiplying y_comp by size
        let y_floor = f32.floor(halfsize+focusPoint.2)
        let y_comp =
          if (y_step_dir == -1f32 && focusPoint.2 - f32.floor(focusPoint.2) == 0f32)
          then y_floor - 1f32
          else y_floor
        let x_comp= f32.floor(halfsize+focusPoint.1)
        let v = x_comp+size*y_comp
        --ONE GREATER WHY?
        let index = t32(v)

        --compute the distances using the difference travelled along an axis to the next whole number and the slope or inverse slope
        let dy = if vertical then 1f32 else if horizontal then 0f32 else (anchorX-focusPoint.1)*slope
        let dx = if vertical then 0f32 else if horizontal then 1f32 else (anchorY-focusPoint.2)*(1/slope)
        let p_anchor_x = (anchorX, focusPoint.2+dy)
        let p_anchor_y = (focusPoint.1+dx, anchorY)

        let dist_p_x = distance focusPoint p_anchor_x
        let dist_p_y = distance focusPoint p_anchor_y

        in
          if horizontal then
            unsafe let A[write_index] = (dist_p_x, index)
            in (A, p_anchor_x, anchorX + 1f32, anchorY, write_index+1)
          else if vertical then
            unsafe let A[write_index] = (dist_p_x, index)
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

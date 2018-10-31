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

  let index (focusPoint: point) (halfsize: f32) (y_step_dir): i32 =
      let y_floor = f32.floor(halfsize+focusPoint.2)
      let y_comp =
        if (y_step_dir == -1f32 && focusPoint.2 - f32.floor(focusPoint.2) == 0f32)
        then y_floor - 1f32
        else y_floor
      let x_comp= f32.floor(halfsize+focusPoint.1)
      in t32(x_comp+(halfsize*2f32)*y_comp)

  -- let nextpointonline (vertical: bool) (horizontal: bool) (anchorX: point) (anchorY: point) (slope: f32) (focusPoint: point): point or array of points and lengths
  let nextpointonline (slope: f32) (vertical: bool) (focusPoint: point): point =
      let y_step_dir = if slope < 0f32 then -1f32 else 1f32
      let anchorX = if vertical then focusPoint.1 else f32.floor(focusPoint.1) + 1f32
      let anchorY = if slope == 0 then focusPoint.2 else if y_step_dir == -1f32
             then f32.ceil(focusPoint.2) - 1f32
             else f32.floor(focusPoint.2) + 1f32
      let dy = if slope == 1 then 1f32 else if slope == 0 then 0f32 else (anchorX-focusPoint.1)*slope
      let dx = if slope == 1 then 0f32 else if slope == 0 then 1f32 else (anchorY-focusPoint.2)*(1/slope)
      let p_anchor_x = (anchorX, focusPoint.2+dy)
      let p_anchor_y = (focusPoint.1+dx, anchorY)
      in if p_anchor_x.1 < p_anchor_y.1 then p_anchor_x else p_anchor_y

  let getFocusPoints (entryPoint: point) slope vertical halfsize y_step_dir =
    -- let w = 0
    let A = replicate (t32(2f32*halfsize*2f32-1f32)) (f32.lowest, f32.lowest)
    let (A, _, _) =
      loop (A, focusPoint, write_index) = (A, entryPoint, 0)
      while ( isInGrid halfsize y_step_dir focusPoint ) do
        let nextpoint = (nextpointonline slope vertical focusPoint)
        in unsafe let A[write_index] = focusPoint
        in (A, nextpoint, write_index+1)
    in A

  let lengths
     (grid_size: i32)
     (sint: f32)
     (cost: f32)
     (entryPoint: point): [](f32, i32) =

     let vertical = f32.abs(cost) == 1
     let slope =  cost/(-sint)

     let size = r32(grid_size)
     let halfsize = size/2.0f32

     let y_step_dir = if slope < 0f32 then -1f32 else 1f32
     let focuspoints = (getFocusPoints entryPoint slope vertical halfsize y_step_dir)
     let mf = map(\i ->
          let ind = if !(isInGrid halfsize y_step_dir focuspoints[i]) then -1 else index focuspoints[i] halfsize y_step_dir
          let dist = (unsafe (distance focuspoints[i] focuspoints[i+1]))
          in (dist, ind)
      ) (iota (length focuspoints))
      in mf
}

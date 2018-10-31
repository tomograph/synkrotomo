-- ==
-- input@../data/matrixinputf32rad128
-- input@../data/matrixinputf32rad256
import "line_lib"
open Lines

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
     let A = replicate (t32(2f32*halfsize*2f32-1f32)) (f32.lowest, f32.lowest)
     let (A, _, _) =
     loop (A, focusPoint, write_index) = (A, entryPoint, 0)
          while ( isInGrid halfsize y_step_dir focusPoint ) do
          let nextpoint = (nextpointonline slope vertical focusPoint)
          in unsafe let A[write_index] = focusPoint
          in (A, nextpoint, write_index+1)
     in A

let lengths_map
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

let weights_map   (angles: []f32)
               (rays: []f32)
               (gridsize: i32) : [][](f32,i32) =
     let halfsize = r32(gridsize)/2
     let entrypoints = convert2entry angles rays halfsize
     in map (\(p,sc) -> (lengths_map gridsize sc.1 sc.2 p)) entrypoints

let main  (angles: []f32)
          (rays: []f32)
          (gridsize: i32) : [][](f32,i32) =
     weights_map angles rays gridsize

-- ==
-- input@../data/matrixinputf32rad128
-- input@../data/matrixinputf32rad256
import "line_lib"
open Lines

let lengths    (grid_size: i32)
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
        let index = t32(x_comp+size*y_comp)

        --compute the distances using the difference travelled along an axis to the
        --next whole number and the slope or inverse slope
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

let weights_jh   (angles: []f32)
              (rays: []f32)
              (gridsize: i32) : [][](f32,i32) =
    let halfsize = r32(gridsize)/2
    let entrypoints = convert2entry angles rays halfsize
    in map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints

let main  (angles: []f32)
         (rays: []f32)
         (gridsize: i32) : [][](f32,i32) =
    weights_jh angles rays gridsize

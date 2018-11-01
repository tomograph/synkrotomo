import "line_lib"
open Lines


-- function which computes the weight of pixels in grid_column for ray with entry/exit p
module Matrix =
{
     --- DOUBLE PARALLEL
     let calculate_weight(ent: point)
                         (ext: point)
                         (i: i32)
                         (N: i32) : [](f32,i32) =
          let Nhalf = N/2
          -- handle all lines as slope < 1 reverse the others
          let slope = (ext.2 - ent.2)/(ext.1 - ent.1)
          let reverse = slope > 1
          let k = if reverse then 1/slope else slope
          let gridentry = if reverse then (ent.2,ent.1) else ent
          --- calculate stuff
          let ymin = k*(r32(i) - gridentry.1) + gridentry.2 + r32(Nhalf)
          let yplus = k*(r32(i) + 1 - gridentry.1) + gridentry.2 + r32(Nhalf)
          let Ypixmin = t32(f32.floor(ymin))
          let Ypixplus = t32(f32.floor(yplus))
          let baselength = f32.sqrt(1+k ** 2.0f32)
          -- in [(t32(Ypixmin),baselength),(t32(Ypixplus),baselength)]
          let Ypixmax = i32.max Ypixmin Ypixplus
          let ydiff = yplus - ymin
          let yminfact = (r32(Ypixmax) - ymin)/ydiff
          let yplusfact = (yplus - r32(Ypixmax))/ydiff
          let lymin = yminfact*baselength
          let lyplus = yplusfact*baselength
          let iindex = i+Nhalf
          let pixmin = if reverse then iindex*N+Ypixmin else iindex+Ypixmin*N
          let pixplus = if reverse then iindex*N+Ypixplus else iindex+Ypixplus*N
          let min = if (pixmin >= 0 && pixmin < N ** 2) then
               (if Ypixmin == Ypixplus then (baselength,pixmin) else (lymin,pixmin))
               else (-1f32,-1i32)
          let plus = if (pixplus >= 0 && pixplus < N ** 2) then
               (if Ypixmin == Ypixplus then (-1f32,-1i32) else (lyplus,pixplus))
               else (-1f32,-1i32)
          in [min,plus]

     -- assuming flat lines and gridsize even
     let weights_doublepar    (angles: []f32)
                              (rays: []f32)
                              (gridsize: i32): [][](f32,i32) =
          let halfgridsize = gridsize/2
          let entryexitpoints =  convert2entryexit angles rays (r32(halfgridsize))
          in map(\(ent,ext) -> (flatten(map (\i ->
                    calculate_weight ent ext i gridsize
               )((-halfgridsize)...(halfgridsize-1))))) entryexitpoints

     --- JH VERSION
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

     ------------------------MAP----------------------------------------------------------
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
}

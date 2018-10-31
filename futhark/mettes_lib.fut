-- ==
-- compiled input {
--    -2f32
--    -0.5f32
--    2f32
--    0.5f32
--    0i32
--    4i32
-- }
-- output {
--   [10i32, -1i32]
--   [10i32,-1i32]
--   [1.030776f32,-1f32]
-- }
-- compiled input {
--    -1f32
--    2f32
--    2f32
--    0f32
--    0i32
--    4i32
-- }
-- output {
--   [10i32,2i32]
--   [14i32,2i32]
--   [0.600925f32,0.600925f32]
-- }
module Mette = {
     type point  = ( f32, f32 )

     let deg2rad(deg: f32): f32 = (f32.pi / 180f32) * deg

     -- find x given y
     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if cost == 0 then ray else (ray-y*sint)/cost

     -- find y given x
     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if sint == 0 then ray else (ray-x*cost)/sint

     -- Degrees to radians converter
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint)
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let p_top = (find_x maxval ray cost sint, maxval)
          let p_right = (maxval, find_y maxval ray cost sint)
          let ent = if f32.abs(p_left.2) <= maxval && sint != 0 then p_left else if p_bottom.1 <= p_top.1 then p_bottom else p_top
          let ext = if f32.abs(p_right.2) <= maxval && sint != 0 then p_right else if p_bottom.1 <= p_top.1 then p_top else p_bottom
          in (ent, ext)

     -- convertion to sin/cos arrays of array of degrees
     let convert2sincos (angles: []f32) : []point =
          map (\t -> (f32.sin(deg2rad(t)),f32.cos(deg2rad(t)))) angles

     --convert to entry points
     let convert2entryexit (angles: []f32) (rays: []f32) (maxval: f32): [] (point, point) =
          let sincos = convert2sincos angles
          let anglesrays = flatten(map (\t -> map(\r -> (t.1,t.2,r)) rays) sincos)
          in map(\(s,c,r) -> entryexitPoint s c r maxval) anglesrays

     -- function which computes the weight of pixels in grid_column for ray with entry/exit p
     let calculate_weight(ent: point)
               (ext: point)
               (i: i32)
               (N: i32) : [](i32,f32) =
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
               (if Ypixmin == Ypixplus then (pixmin,baselength) else (pixmin,lymin))
               else (-1i32,-1f32)
          let plus = if (pixplus >= 0 && pixplus < N ** 2) then
               (if Ypixmin == Ypixplus then (-1i32,-1f32) else (pixplus,lyplus))
               else (-1i32,-1f32)
          in [min,plus]

     -- assuming flat lines and gridsize even
     let compute_weights(angles: []f32) (rays: []f32) (gridsize: i32): [][](i32,f32) =
          let halfgridsize = gridsize/2
          let entryexitpoints =  convert2entryexit angles rays (r32(halfgridsize))
          in map(\(ent,ext) -> (flatten(map (\i ->
                    calculate_weight ent ext i gridsize
               )((-halfgridsize)...(halfgridsize-1))))) entryexitpoints

     let notSparseMatMult [num_rows] [num_cols]
                          (mat_vals : [num_rows][num_cols](i32,f32))
                          (vect : []f32) : [num_rows]f32 =
         map (\row -> reduce (+) 0 <| map (\(ind, v) -> unsafe (if ind == -1 then 0.0 else v*vect[ind]) ) row ) mat_vals

     let forwardprojection(angles: []f32) (rays: []f32) (gridsize: i32) (vector: []f32) : []f32 =
          let weights = compute_weights angles rays gridsize
          in notSparseMatMult weights vector



}

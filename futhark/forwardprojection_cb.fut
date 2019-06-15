-- compiled input {
--  [0.0f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.03100960116f32, 2.03100960116f32, 2.03100960116f32, 2.03100960116f32] }
-- compiled input {
--  [3.14159265359f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.03100960116f32, 2.03100960116f32, 2.03100960116f32, 2.03100960116f32] }
-- compiled input {
--  [1.57079632679f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.03100960116f32, 2.03100960116f32, 2.03100960116f32, 2.03100960116f32] }
-- compiled input {
--  [4.71238898038f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.03100960116f32, 2.03100960116f32, 2.03100960116f32, 2.03100960116f32] }
-- compiled input {
--  [0.78539816339f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.40206112814f32, 2.40206112814f32, 2.40206112814f32, 2.40206112814f32] }
-- compiled input {
--  [2.35619449019f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.40206112814f32, 2.40206112814f32, 2.40206112814f32, 2.40206112814f32] }
-- compiled input {
--  [3.92699081699f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.40206112814f32, 2.40206112814f32, 2.40206112814f32, 2.40206112814f32] }
-- compiled input {
--  [5.49778714378f32]
--  2.0f32
--  2.0f32
--  [1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32,1.0f32]
-- 2
-- 2
-- }
-- output { [2.40206112814f32, 2.40206112814f32, 2.40206112814f32, 2.40206112814f32] }
-- ==
-- input@data/fpcbinputf32rad64
-- input@data/fpcbinputf32rad128
-- input@data/fpcbinputf32rad256


module fplib = {
     type pointXYZ  = ( f32, f32, f32 )

     let find_detector_points (cos: f32) (sin: f32) (initial_points: []pointXYZ) (detector_row_count: i32) (detector_col_count: i32): [](pointXYZ, i32) =
          let (_,_,zs) = unzip3 initial_points
          in flatten <| map(\col ->
             let (x,y,_) = unsafe initial_points[col]
             let detector_x = replicate detector_row_count (cos*x-sin*y)
             let detector_y = replicate detector_row_count (sin*x+cos*y)
             let indexes = map(\row -> col+row*detector_col_count)(iota detector_row_count)
             let points = zip3 detector_x detector_y zs
             in zip points indexes
          )(iota detector_col_count)

     let swap_xy (point: pointXYZ) : pointXYZ =
          (-point.2,-point.1,point.3)

     let get_partitioning_bools (source: pointXYZ) (detector: pointXYZ): bool =
          -- difference in x values
          let norm_x = f32.abs(source.1-detector.1)
          -- different in y values
          let norm_y = f32.abs(source.2-detector.2)
          -- if diff in y less than diff in x then line is flat
          let ysmall = norm_y < norm_x
          in ysmall

     let preprocess [a](angles: [a]f32) (origin_source_dist: f32) (origin_detector_dist: f32) (detector_row_count: i32) (detector_col_count: i32) (detector_z_offset: i32): ([](pointXYZ, pointXYZ), [](pointXYZ, pointXYZ), []bool, []i32)=
          let halfwidth = (r32(detector_col_count-1))/2.0f32
          let halfheight = (r32(detector_row_count-1))/2.0f32
          let x_vals = replicate detector_col_count origin_detector_dist
          let y_vals = map(\row -> halfwidth-(r32(row)))(iota detector_col_count)
          let z_vals = map(\row -> halfheight-(r32(row))+(r32(detector_z_offset)))(iota detector_row_count)
          let xyz_points = zip3 x_vals y_vals z_vals
          let source_detector = flatten <| map(\angle_index ->
               let angle = angles[angle_index]
               let sin = f32.sin(angle)
               let cos = f32.cos(angle)
               -- source location
               let source =  (-origin_source_dist*cos, -origin_source_dist*sin, 0.0f32)
               -- points where rays hit detector
               let detector_points = find_detector_points cos sin xyz_points detector_row_count detector_col_count
               in map(\(detector_point, detector_index)->
                    -- determine if x, y or z diff is smallest
                    let ysmall = get_partitioning_bools source detector_point
                    in (source, detector_point, ysmall, (detector_index+detector_col_count*detector_row_count*angle_index))
               )detector_points
          ) (iota a)
          -- partition into y smallest, z smallest, x smallest
          let parts = partition (\(_,_,bysmall,_) -> bysmall) source_detector
          -- transpose the lines
          let ysmallest = map(\(source,detector,_,_)->((swap_xy(source)),(swap_xy(detector))))parts.1
          let xsmallest = map(\(source,detector,_,_)->(source,detector))parts.2
          -- save original indexes and transposition state
          let (_,_,bysmall,projection_indexes) = unzip4 (parts.1 ++ parts.2)
          in (ysmallest, xsmallest, bysmall, projection_indexes)

     let transpose_xy [n](volume: [n][n][n]f32): [n][n][n]f32 =
          map(\i-> transpose volume[i,0:n,0:n])(iota n)

     let fix_projections [p](proj:[p]f32) (ysmall:[p]bool):([]f32,[]f32) =
          let zipped = zip proj ysmall
          let parts = partition (\(_,ysml) -> ysml )zipped
          let (ysmallest, _) = unzip parts.1
          let (xsmallest, _) = unzip parts.2
          in (ysmallest, xsmallest)

     let postprocess_fp [p][x][y](projection_indexes: [p]i32) (ysmall: [x]f32) (xsmall: [y]f32): [p]f32 =
          scatter (replicate (x+y) 0.0f32) projection_indexes (ysmall ++ xsmall)

     let false_to_zero (b1: bool) (b2: bool): f32 =
          if b1 && b2 then 1.0f32 else 0.0f32

     let intersection_ratio (bot: f32) (top: f32) (halfsize: i32): (f32, bool, bool, bool) =
          let pixbot = t32(f32.floor(bot))
          let pixtop = t32(f32.floor(top))
          let singlepix = pixbot == pixtop
          let pixmax = r32(i32.max pixbot pixtop)
          let diff = top-bot
          -- in case of single voxel situation this could be -1... check for 2d case too
          let ratio = (pixmax - bot)/diff
          let botwithinbounds = pixbot >= -halfsize && pixbot < halfsize
          let topwithinbounds = pixtop >= -halfsize && pixtop < halfsize
          in (ratio, botwithinbounds, topwithinbounds, singlepix)

     let pixel_index (x: i32) (y: i32) (z: i32) (N: i32): i32 =
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          let i = halfsize-1-y
          let j = x+halfsize
          let k = halfsize-1-z
          in i + j*N + k*N*N

     let get_value (i: i32) (p1: pointXYZ) (p2: pointXYZ) (N: i32) (volume: []f32) : f32 =
          -- increase in x pr. y
          let slope_y = (p2.1-p1.1)/(p2.2-p1.2)
          -- increase in z value pr. 1 unit y
          let slope_z_tmp = (p2.3-p1.3)*f32.sqrt(1+slope_y**2.0f32)/f32.sqrt((p2.1-p1.1)**2.0f32+(p2.2-p1.2)**2.0f32)
          let slope_z = if p2.2<p1.2 then -slope_z_tmp else slope_z_tmp
          -- intercept of line x = slope_y*y+intercept_y
          let intercept_y = p1.1-slope_y*p1.2
          -- we now have dy = 1, dx = slope_y, dz = slope_z
          -- intercept of line z = slope_z*l*y+intercept_z
          let intercept_z = p1.3-slope_z*p1.2
          -- the z-value when y = i
          let zminus = r32(i) * slope_z+intercept_z
          -- the z-value when y = i+1
          let zplus = zminus + slope_z
          -- the x value when y = i
          let xminus = slope_y * r32(i) + intercept_y
          -- the x-value when y = i+1
          let xplus = xminus + slope_y
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          -- determine ratios and whether each intersection is within bounds
          let (r_x, bounds_bot_x, bounds_top_x, single_x) = intersection_ratio xminus xplus halfsize
          let (r_z, bounds_bot_z, bounds_top_z, single_z) = intersection_ratio zminus zplus halfsize

          -- determine length of line through a voxel if it only passes through this one voxel.<
          let lbase = f32.sqrt(1.0f32+slope_y**2.0f32+slope_z**2.0f32)

          -- get pixel values (seem correct)
          let pixbotbot = unsafe volume[i32.min (pixel_index (t32(f32.floor(xminus))) i (t32(f32.floor(zminus))) N) (N**3-1)]
          let pixbottop = if (single_x && !single_z) || r_x > r_z then unsafe volume[i32.min (pixel_index (t32(f32.floor(xminus))) i (t32(f32.floor(zplus))) N) (N**3-1)] else 0.0
          let pixtopbot = if (single_z && !single_x) || r_x < r_z then unsafe volume[i32.min (pixel_index (t32(f32.floor(xplus))) i (t32(f32.floor(zminus))) N) (N**3-1)] else 0.0
          let pixtoptop = if (!single_z && !single_x) then unsafe volume[i32.min (pixel_index (t32(f32.floor(xplus))) i (t32(f32.floor(zplus))) N) (N**3-1)] else 0.0

          -- make values outside bounds be zero
          let botbot = false_to_zero bounds_bot_x bounds_bot_z
          let bottop = false_to_zero bounds_bot_x bounds_top_z
          let topbot = false_to_zero bounds_top_x bounds_bot_z
          let toptop = false_to_zero bounds_top_x bounds_top_z

          -- only one intersecting voxel
          let r_single = botbot*pixbotbot
          -- one intersecting voxels
          let r_double_x = r_z*pixbotbot+(1-r_z)*pixbottop
          let r_double_z = r_x*pixbotbot+(1-r_x)*pixtopbot
          -- threee intersecting voxels
          let r_one = r_z*botbot*pixbotbot+(r_x-r_z)*bottop*pixbottop+(1-r_x)*toptop*pixtoptop
          let r_two = r_x*botbot*pixbotbot+(r_z-r_x)*topbot*pixtopbot+(1-r_z)*toptop*pixtoptop
          let intersectsum = if single_x && single_z then r_single
               else if single_x then r_double_x
               else if single_z then r_double_z
               else if r_x > r_z then r_one
               else r_two
          in lbase*intersectsum

     let fp (points: [](pointXYZ, pointXYZ)) (N: i32) (volume: []f32): []f32 =
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          -- ent, ext points seem correct.
          in map(\(ent, ext)->
                         reduce (+) 0.0f32 <| map(\(i) ->
                              get_value i ent ext N volume
                         )((-halfsize)...(halfsize-1))
                    )points
}

open fplib

let main  [n][a] (angles : *[a]f32)
          (origin_source_dist: f32)
          (origin_detector_dist: f32)
          (volume : *[n]f32)
          (detector_row_count: i32)
          (detector_col_count: i32)
          (detector_z_offset: i32)
          (N : i32): []f32 =
          let (y_small, x_small, _, projection_indexes) = preprocess angles origin_source_dist origin_detector_dist detector_row_count detector_col_count detector_z_offset

          let imageTxy =  if (N < 10000)
                        then flatten_3d <| transpose_xy <|copy (unflatten_3d N N N volume)
                        else (replicate n 1.0f32)

          let fpxsmall = fp x_small N volume
          let fpysmall = fp y_small N imageTxy
          -- it seems projection indexes might be wrong!
          in postprocess_fp projection_indexes fpysmall fpxsmall

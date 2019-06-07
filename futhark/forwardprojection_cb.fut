-- ==
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


module fplib = {
     type pointXYZ  = ( f32, f32, f32 )

     let find_detector_points (sin: f32) (cos: f32) (origin_detector_dist: f32) (detector_size: i32): [](pointXYZ, i32) =
          let halfsize = (r32(detector_size-1))/2.0f32
          let z_vals = reverse <| map(\i -> -halfsize+(r32(i)))(iota detector_size)
          in flatten <| map(\col ->
             let detector_x = replicate detector_size (origin_detector_dist*cos - sin * (halfsize-(r32(col))))
             let detector_y = replicate detector_size (origin_detector_dist*sin + cos * (halfsize-(r32(col))))
             let indexes = map(\row -> col+row*detector_size)(iota detector_size)
             let points = zip3 detector_x detector_y z_vals
             in zip points indexes
          )(iota detector_size)

     let swap_xy (point: pointXYZ) : pointXYZ =
          (point.2,point.1,point.3)

     let get_partitioning_bools (source: pointXYZ) (detector: pointXYZ): bool =
          -- difference in x values
          let norm_x = f32.abs(source.1-detector.1)
          -- different in y values
          let norm_y = f32.abs(source.2-detector.2)
          -- if diff in y less than diff in x then line is flat
          let bswap_xy = norm_y < norm_x
          in bswap_xy

     let preprocess [a](angles: [a]f32) (origin_source_dist: f32) (origin_detector_dist: f32) (detector_size: i32): ([](pointXYZ, pointXYZ), [](pointXYZ, pointXYZ), []bool, []i32)=
          let source_detector = flatten <| map(\angle_index ->
               let angle = angles[angle_index]
               let cos = f32.cos(angle)
               let sin = f32.sin(angle)
               -- source location
               let source =  (-origin_source_dist*cos, -origin_source_dist*sin, 0.0f32)
               -- points where rays hit detector
               let detector_points = find_detector_points sin cos origin_detector_dist detector_size
               in map(\(detector_point, detector_index)->
                    -- determine if x, y or z diff is smallest
                    let bswap_xy = get_partitioning_bools source detector_point
                    in (source, detector_point, bswap_xy, (detector_index+detector_size*detector_size*angle_index))
               )detector_points
          ) (iota a)
          -- partiition into y smallest, z smallest, x smallest
          let parts = partition (\(_,_,bswap_xy,_) -> bswap_xy) source_detector
          -- transpose the lines
          let transpose_xy = map(\(source,detector,_,_)->((swap_xy source), (swap_xy detector)))parts.1
          let transpose_no = map(\(source,detector,_,_)->(source,detector))parts.2
          -- save original indexes and transposition state
          let (_,_,bswap_xy,projection_indexes) = unzip4 (parts.2 ++ parts.1)
          in (transpose_no, transpose_xy, bswap_xy, projection_indexes)


     let transpose_xz [n](volume: [n][n][n]f32): [n][n][n]f32 =
          map(\i-> transpose volume[0:n,0:n,i])(iota n)

     let transpose_xy [n](volume: [n][n][n]f32): [n][n][n]f32 =
          map(\i-> transpose volume[i,0:n,0:n])(iota n)

     let fix_projections [p](proj:[p]f32) (bswap_xy:[p]bool):([]f32,[]f32) =
          let zipped = zip proj bswap_xy
          let parts = partition (\(_,xy) -> xy )zipped
          let (xy, _) = unzip parts.1
          let (no, _) = unzip parts.2
          in (no, xy)

     let postprocess_fp [p][x][y](projection_indexes: [p]i32) (val_no: [x]f32) (val_xy: [y]f32): [p]f32 =
          scatter (replicate (x+y) 0.0f32) projection_indexes (val_no ++ val_xy)

     let false_to_zero (b1: bool) (b2: bool): f32 =
          if b1 && b2 then 1.0f32 else 0.0f32

     let intersection_ratio (xbot: f32) (xtop: f32) (halfsize: i32): (f32, bool, bool) =
          let Xpixbot = t32(f32.floor(xbot))
          let Xpixtop = t32(f32.floor(xtop))
          let pixmax = r32(i32.max Xpixbot Xpixtop)
          let diff = xtop-xbot
          -- in case of single voxel situation this could be -1... check for 2d case too
          let ratio = (pixmax - xbot)/diff
          let xbotwithinbounds = Xpixbot >= -halfsize && Xpixbot < halfsize
          let xtopwithinbounds = Xpixtop >= -halfsize && Xpixtop < halfsize
          in (ratio, xbotwithinbounds, xtopwithinbounds)

     let pixel_index (x: f32) (y: i32) (z: f32) (N: i32): i32 =
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          let i = t32(f32.floor(x))+halfsize
          let j = halfsize-1-y
          let k = halfsize-1-t32(f32.floor(z))
          in i + j*N + k*N*N

     -- using right handed coordinate system
     let get_value (i: i32) (p1: pointXYZ) (p2: pointXYZ) (N: i32) (volume: []f32) : f32 =
          -- increase in x pr. y
          let slope_y = (p2.1-p1.1)/(p2.2-p1.2)
          -- increase in z value pr. 1 unit y
          let slope_z = (p2.3-p1.3)*f32.sqrt(1+slope_y**2.0f32)/f32.sqrt((p2.1-p1.1)**2.0f32+(p2.2-p1.2)**2.0f32)
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
          let (r_x, bounds_bot_x, bounds_top_x) = intersection_ratio xminus xplus halfsize
          let (r_z, bounds_bot_z, bounds_top_z) = intersection_ratio xminus xplus halfsize

          -- determine length of line through a voxel if it only passes through this one voxel.<
          let lbase = f32.sqrt(1.0f32+slope_y**2.0f32+slope_z**2.0f32)

          -- get pixel values (seem correct)
          let pixbotbot = unsafe volume[i32.min (pixel_index xminus i zminus N) (N**3-1)]
          let pixbottop = unsafe volume[i32.min (pixel_index xminus i zplus N) (N**3-1)]
          let pixtopbot = unsafe volume[i32.min (pixel_index xplus i zminus N) (N**3-1)]
          let pixtoptop = unsafe volume[i32.min (pixel_index xplus i zplus N) (N**3-1)]

          -- make values outside bounds be zero
          let botbot = false_to_zero bounds_bot_x bounds_bot_z
          let bottop = false_to_zero bounds_bot_x bounds_top_z
          let topbot = false_to_zero bounds_top_x bounds_bot_z
          let toptop = false_to_zero bounds_top_x bounds_top_z

          let r_one = r_z*botbot*pixbotbot+(r_x-r_z)*bottop*pixbottop+(1-r_x)*toptop*pixtoptop
          let r_two = r_x*botbot*pixbotbot+(r_z-r_x)*topbot*pixtopbot+(1-r_z)*toptop*pixtoptop
          in if r_x > r_z then r_one*lbase else r_two*lbase

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
          (detector_size: i32)
          (N : i32): []f32 =
          let (x_small, y_small, _, projection_indexes) = preprocess angles origin_source_dist origin_detector_dist detector_size

          let imageTxy =  if (N < 10000)
                        then flatten_3d <| transpose_xy <| copy (unflatten_3d N N N volume)
                        else (replicate n 1.0f32)

          let fpxsmall = fp x_small N volume
          let fpysmall = fp y_small N imageTxy

          in postprocess_fp projection_indexes fpxsmall fpysmall

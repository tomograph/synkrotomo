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

     let swap_xz (point: pointXYZ) : pointXYZ =
          (point.3,point.2,point.1)

     let get_partitioning_bools (source: pointXYZ) (detector: pointXYZ): (bool, bool) =
          let norm_x = f32.abs(source.1-detector.1)
          let norm_y = f32.abs(source.2-detector.2)
          let norm_z = f32.abs(source.3-detector.3)
          let bswap_xy = norm_y < norm_x && norm_y <= norm_z --y smallest
          let bswap_xz = norm_z < norm_x && norm_z <= norm_y --z smallest
          -- otherwise x is smallest - great!
          in (bswap_xy, bswap_xz)

     let preprocess [a](angles: [a]f32) (origin_source_dist: f32) (origin_detector_dist: f32) (detector_size: i32): ([](pointXYZ, pointXYZ), [](pointXYZ, pointXYZ), [](pointXYZ, pointXYZ), [](bool,bool), []i32)=
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
                    let (bswap_xy, bswap_xz) = get_partitioning_bools source detector_point
                    in (source, detector_point, bswap_xy, bswap_xz, (detector_index+detector_size*detector_size*angle_index))
               )detector_points
          ) (iota a)
          -- partiition into y smallest, z smallest, x smallest -CHECK this!
          let parts = partition2 (\(_,_,bswap_xy,_,_) -> bswap_xy) (\(_,_,_,bswap_xz,_) -> bswap_xz) source_detector
          -- transpose the lines
          let transpose_xy = map(\(source,detector,_,_,_)->((swap_xy source), (swap_xy detector)))parts.1
          let transpose_xz = map(\(source,detector,_,_,_)->((swap_xz source), (swap_xz detector)))parts.2
          let transpose_no = map(\(source,detector,_,_,_)->(source,detector))parts.3
          -- save original indexes and transposition state
          let (_,_,bswap_xy,bswap_xz,projection_indexes) = unzip5 (parts.3 ++ parts.1 ++ parts.2)
          in (transpose_no, transpose_xy, transpose_xz, (zip bswap_xy bswap_xz), projection_indexes)


     let transpose_xz [n](volume: [n][n][n]f32): [n][n][n]f32 =
          map(\i-> transpose volume[0:n,0:n,i])(iota n)

     let transpose_xy [n](volume: [n][n][n]f32): [n][n][n]f32 =
          map(\i-> transpose volume[i,0:n,0:n])(iota n)

     let fix_projections [p](proj:[p]f32) (bswap_xy_xz:[p](bool,bool)):([]f32,[]f32,[]f32) =
          let (bswap_xy, bswap_xz) = unzip bswap_xy_xz
          let zipped = zip3 proj bswap_xy bswap_xz
          let parts = partition2 (\(_,xy,_) -> xy ) (\(_,_,xz) -> xz) zipped
          let (xy, _, _) = unzip3 parts.1
          let (xz, _, _) = unzip3 parts.2
          let (no, _, _) = unzip3 parts.3
          in (no, xy, xz)

     let postprocess_fp [p][x][y][z](projection_indexes: [p]i32) (val_no: [x]f32) (val_xy: [y]f32) (val_xz: [z]f32): [p]f32 =
          --val_no ++ (replicate y 0.0f32) ++ (replicate z 0.0f32)
          scatter (replicate (x+y+z) 0.0f32) projection_indexes (val_no ++ val_xy ++ val_xz)

     let false_to_zero (b1: bool) (b2: bool): f32 =
          if b1 && b2 then 1.0f32 else 0.0f32

     let intersection_ratio (xbot: f32) (xtop: f32) (halfsize: i32): (f32, bool, bool) =
          let Xpixbot = t32(f32.floor(xbot))
          let Xpixtop = t32(f32.floor(xtop))
          let pixmax = r32(i32.max Xpixbot Xpixtop)
          let diff = xtop-xbot
          let ratio = (pixmax - xbot)/diff
          let xbotwithinbounds = Xpixbot >= -halfsize && Xpixbot < halfsize
          let xtopwithinbounds = Xpixtop >= -halfsize && Xpixtop < halfsize
          in (ratio, xbotwithinbounds, xtopwithinbounds)

     let pixel_index (x: i32) (y: f32) (z: f32) (N: i32): i32 =
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          let i = x+halfsize
          let j = halfsize-t32(f32.floor(y))-1
          let k = halfsize-t32(f32.floor(z))-1
          in i + j*N + k*N*N

     -- using right handed coordinate system
     let get_value (i: i32) (p1: pointXYZ) (p2: pointXYZ) (N: i32) (volume: []f32) : f32 =
          let slope_y = (p2.1-p1.1)/(p2.2-p1.2)
          let intercept_y = p1.1 - slope_y*p1.2
          let slope_z = (p2.1-p1.1)/(p2.3-p1.3)
          let intercept_z = p1.1 - slope_z*p1.3
          let lbase = f32.sqrt(1.0f32+slope_y**2.0f32+slope_z**2.0f32)
          let xbot_y = r32(i)*slope_y + intercept_y
          let xtop_y = xbot_y+slope_y
          let xbot_z = r32(i)*slope_z + intercept_z
          let xtop_z = xbot_z+slope_z
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
          let (r_y, bounds_bot_y, bounds_top_y) = intersection_ratio xbot_y xtop_y halfsize
          let (r_z, bounds_bot_z, bounds_top_z) = intersection_ratio xbot_z xtop_z halfsize

          let pixbotbot = unsafe volume[i32.min (pixel_index i xbot_y xbot_z N) (N**3-1)]
          let pixbottop = unsafe volume[i32.min (pixel_index i xbot_y xtop_z N) (N**3-1)]
          let pixtopbot = unsafe volume[i32.min (pixel_index i xtop_y xbot_z N) (N**3-1)]
          let pixtoptop = unsafe volume[i32.min (pixel_index i xtop_y xtop_z N) (N**3-1)]

          let botbot = false_to_zero bounds_bot_y bounds_bot_z
          let bottop = false_to_zero bounds_bot_y bounds_top_z
          let topbot = false_to_zero bounds_top_y bounds_bot_z
          let toptop = false_to_zero bounds_top_y bounds_top_z

          let r_one = r_z*botbot*pixbotbot+(r_y-r_z)*bottop*pixbottop+(1-r_y)*toptop*pixtoptop
          let r_two = r_y*botbot*pixbotbot+(r_z-r_y)*topbot*pixtopbot+(1-r_z)*toptop*pixtoptop
          --in if r_y > r_z then r_one*lbase else r_two*lbase
          in lbase

     let fp (points: [](pointXYZ, pointXYZ)) (N: i32) (volume: []f32): []f32 =
          let halfsize = (t32(f32.floor(r32(N)/2.0f32)))
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
          let (x_small, y_small, z_small, _, projection_indexes) = preprocess angles origin_source_dist origin_detector_dist detector_size

          let imageTxy =  if (N < 10000)
                        then flatten_3d <| transpose_xy <| copy (unflatten_3d N N N volume)
                        else (replicate n 1.0f32)

          let imageTxz =  if (N < 10000)
                        then flatten_3d <| transpose_xz <| copy (unflatten_3d N N N volume)
                        else (replicate n 1.0f32)
          -- now run forward porjection on x_small, y_small - but transpose xy, z_small but transpose xz.
          let fpxsmall = fp x_small N volume
          let fpysmall = fp y_small N imageTxy
          let fpzsmall = fp z_small N imageTxz

          in postprocess_fp projection_indexes fpxsmall fpysmall fpzsmall

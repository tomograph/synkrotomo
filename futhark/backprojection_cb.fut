import "box_ray_intersect"
import "vspace"
import "vector"
import "forwardprojection_cb"
open fplib

module vec3 = mk_vspace_3d f32

module bplib = {

     let fact = f32.cos(f32.pi/4)*f32.sqrt(2.0f32)/2.0f32

     let tovec (triple: (f32,f32,f32)) : vec3.vector =
          {x = triple.1, y=triple.2, z=triple.3}

     -- let shadow     (cos: f32)
     --                (sin: f32)
     --                (c: vec3.vector)
     --                (detector_top: i32)
     --                (detector_width: i32)
     --                (source_detector: f32)
     --                (source_origin: f32): [](f32,f32,f32,i32) =
     --      -- unrotate
     --      let (xr,yr) = ((c.x*cos+c.y*sin),(-c.x*sin+c.y*cos))
     --      let detcolmin = t32(f32.floor((yr-fact)*source_detector/(source_origin+xr-fact)))-detector_width/2
     --      let detcolmax = t32(f32.floor((yr+fact)*source_detector/(source_origin+xr-fact)))-detector_width/2
     --      let detrowmin = t32(f32.floor((c.z-fact)*source_detector/(source_origin+xr-fact)))-detector_top
     --      let detrowmax = t32(f32.floor((c.z+fact)*source_detector/(source_origin+xr-fact)))-detector_top
     --
     --      let num_cols = i32.max (detcolmax-detcolmin) 0
     --      let num_rows = i32.max (detrowmax-detrowmin) 0
     --
     --      let det_x = unsafe(replicate num_cols (source_detector-source_origin))
     --      let det_y = unsafe(map(\col -> r32(detcolmin)+0.5f32+r32(col))(iota num_cols))
     --      let det_z = unsafe(map(\row -> r32(detcolmin)+0.5f32+r32(row))(iota num_rows))
     --      --let zs = unsafe(replicate num_cols 0.0f32)
     --      let indexes = unsafe(replicate num_cols 0)
     --      let zs = flatten <| unsafe(replicate num_cols det_z)
     --      -- -- -- rotate back
     --      let unrotated = map(\(x,y)->((cos*x-sin*y),(sin*x+cos*y)))(zip det_x det_y)
     --      let (xs,ys) = unzip <| flatten <| unsafe (replicate num_rows unrotated)
     --      -- let indexes = flatten <| map(\row -> map(\col -> col+row*detector_width)((detcolmin) ... (detcolmax)))((detrowmin) ... (detrowmax))
     --      let xyzi = zip4 xs ys zs indexes
     --      in xyzi

     let bp [p](angles: [](f32, f32)) (detector_top: i32) (detector_row_count: i32) (detector_col_count) (origin_detector: f32) (projections: [p]f32) (z: i32) (r: i32) (c: i32) (vol_top): []f32 =
               let rows = (-r/2) ... (r/2-1)
               let cols = (-c/2) ... (c/2-1)
               let halfwidth = (r32(detector_col_count-1))/2.0f32
               let x_vals = replicate detector_col_count origin_detector
               let y_vals = map(\row -> halfwidth-(r32(row)))(iota detector_col_count)
               let z_vals = map(\row -> (r32(detector_top))-0.5-(r32(row)))(iota detector_row_count)
               let xyz_points = zip3 x_vals y_vals z_vals
               let slices = map(\z -> vol_top+z)(iota z)
               in flatten <| map(\irow ->
                         flatten <| map(\icol ->
                              map(\islice ->
                                   let min_corner = {x=(r32(irow)),y=(r32(icol)), z=(r32(islice))}
                                   in reduce (+) 0.0f32 <| map(\(cos,sin)->
                                        let detector_points = find_detector_points cos sin xyz_points detector_row_count detector_col_count
                                        let source = {x=(-1.0f32*cos), y=(-1.0f32*sin), z=0.0f32}
                                        in (reduce (+) 0.0f32 <| map(\((x,y,z),i) ->
                                             let detector_vec = tovec (x,y,z)
                                             in (box_ray_intersect source detector_vec min_corner)*(unsafe projections[i])
                                        )detector_points)
                                   )angles
                              )slices
                         )rows
                    )cols
}

open bplib

let main  (angles : []f32)
          (origin_detector_dist: f32)
          (z : i32)
          (r : i32)
          (c : i32)
          (volume_top: i32)
          (detector_col_count: i32)
          (detector_row_count: i32)
          (detector_top: i32)
          (projections:[]f32): []f32 =
          let cossin = map(\a->(f32.cos(a),f32.sin(a)))angles
          in bp cossin detector_top detector_row_count detector_col_count origin_detector_dist projections z r c volume_top

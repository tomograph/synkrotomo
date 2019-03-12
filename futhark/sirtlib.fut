

module sirtlib = {
type point  = ( f32, f32 )

      let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

      let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-x*cost)/sint

      let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-y*sint)/cost

      let safe_inverse (value: f32) : f32 =
           if value == 0.0 then 0.0 else 1/value

      let inverse (values: []f32) : []f32 =
           map(\v -> safe_inverse v) values

     -- gets entry and exit point in no particular order. might later consider corners and vertical lines on grid edge
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let flat = is_flat cost sint
          let point1 = if flat then ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint) else (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let point2 = if flat then (maxval, find_y maxval ray cost sint) else (find_x maxval ray cost sint, maxval)
          in (point1, point2)

     -- divides data into flat and steep parts
     let fix_projections [p](proj:[p]f32) (is_flat:[p]bool):([]f32,[]f32) =
          let zipped = zip proj is_flat
          let parts = partition(\(_,f) -> f ) zipped
          let (flat, _) = unzip parts.1
          let (steep, _) = unzip parts.2
          in (steep, flat)

     -- reasembles forwardprojection to match input parameter
     let postprocess_fp [p][f][s](projection_indexes: [p]i32) (val_steep: [f]f32) (val_flat: [s]f32): [p]f32 =
          scatter (replicate (f+s) 0.0f32) projection_indexes (val_steep ++ val_flat)

     -- divides in flat and steep and transposes lines
     let preprocess [a](angles: [a]f32) (numrhos: i32): ([](f32, f32, f32), [](f32, f32, f32), []bool, []i32) =
     let cossin = map(\i ->
          let angle = angles[i]
          let cos= f32.cos(angle)
          let sin = f32.sin(angle)
          let lsteep = f32.sqrt(1.0+(sin/cos)**2.0f32)
          let lflat = f32.sqrt(1.0+(cos/sin)**2.0f32)
          let flat = is_flat cos sin
          in (cos, sin, lflat, lsteep, flat, i)
     ) (iota a)
     let is_flat = flatten <| map(\(_,_,_,_,f,_) -> (replicate numrhos f))cossin
     let flat_steep = partition(\(_,_,_,_,f,_) -> f ) cossin
    -- transpose flat lines to make them steep
     let lines = (map(\(cos,sin,_,lsteep,_,_)-> (cos,-sin,lsteep)) flat_steep.2, map (\(cos,sin,lflat,_,_,_)-> (-sin, cos, lflat)) flat_steep.1)
     let angle_indexes = (map (\(_,_,_,_,_,i)-> i) flat_steep.2) ++ (map (\(_,_,_,_,_,i)-> i) flat_steep.1)
     let projection_indexes = flatten <| map(\i -> map(\r-> i*numrhos + r)(iota numrhos))angle_indexes
     in (lines.1, lines.2, is_flat, projection_indexes)

     let intersect_fact (plus: f32) (minus: f32) (mini: f32) (maxi: f32): f32=
          -- is zero if both values are below minimum else the positive difference between minus and yplus
          let b = f32.max (plus-mini) 0.0f32
          -- is zero if both values are above maximum else the positive difference between minus and yplus
          let a = f32.max (maxi-minus) 0.0f32
          -- let l = distance left right
          let d = plus-minus
          let minab = f32.min a b
          let u = if minab == 0.0f32 then 0.0f32 else minab/d
          let fact = f32.min u 1
          in fact

}

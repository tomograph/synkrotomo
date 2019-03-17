

module sirtlib = {
type point  = ( f32, f32 )

      let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

      let safe_inverse (value: f32) : f32 =
           if value == 0.0 then 0.0 else 1/value

      let inverse (values: []f32) : []f32 =
           map(\v -> safe_inverse v) values

     -- divides data into flat and steep parts
     let fix_projections [p](proj:[p]f32) (proj_division:[p]bool):([]f32,[]f32) =
          let zipped = zip proj proj_division
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
     let proj_division = flatten <| map(\(_,_,_,_,f,_) -> (replicate numrhos f))cossin
     let flat_steep = partition(\(_,_,_,_,f,_) -> f ) cossin
    -- transpose flat lines to make them steep
     let lines = (map(\(cos,sin,_,lsteep,_,_)-> (cos,-sin,lsteep)) flat_steep.2, map (\(cos,sin,lflat,_,_,_)-> (-sin, cos, lflat)) flat_steep.1)
     let angle_indexes = (map (\(_,_,_,_,_,i)-> i) flat_steep.2) ++ (map (\(_,_,_,_,_,i)-> i) flat_steep.1)
     let projection_indexes = flatten <| map(\i -> map(\r-> i*numrhos + r)(iota numrhos))angle_indexes
     in (lines.1, lines.2, proj_division, projection_indexes)
}



module sirtlib = {
type point  = ( f32, f32 )

     let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-x*cost)/sint

     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-y*sint)/cost

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
     let postprocess_fp [p][f][s](projection_indexes: [p]f32) (val_steep: [f]f32) (val_flat: [s]f32): [p]f32 =
          scatter (replicate (f+s) 0.0) projection_indexes (val_steep ++ val_flat)

     -- divides in flat and steep and transposes lines
     let preprocess [a](angles: [a]f32) (numrhos: i32): ([](f32, f32, f32), [](f32, f32, f32), []bool, []i32) =
     let numrhos
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
     in (lines.2, lines.1, is_flat, projection_indexes)

     let intersect_steep (i: i32) (ext: point) (ent: point) (Nhalf: i32): ((f32,i32,i32),(f32,i32,i32)) =
          let k = (ext.1 - ent.1)/(ext.2 - ent.2)
          let xmin = k*(r32(i) - ent.2) + ent.1 + (r32(Nhalf))
          let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (r32(Nhalf))
          let Xpixmin = t32(f32.floor(xmin))
          let Xpixplus = t32(f32.floor(xplus))
          let baselength = f32.sqrt(1+k*k)
          let Xpixmax = i32.max Xpixmin Xpixplus
          let xdiff = xplus - xmin
          -- if both equal then l is baselength and we only want one l
          let xminfact = if Xpixmin == Xpixplus then 1 else (r32(Xpixmax) - xmin)/xdiff
          let xplusfact = if Xpixmin == Xpixplus then 0 else (xplus - r32(Xpixmax))/xdiff
          let lxmin = xminfact*baselength
          let lxplus = xplusfact*baselength
          let y = i+Nhalf
          in ((lxmin, Xpixmin, y), (lxplus, Xpixplus, y))

     let calculate_product [n](sin: f32)
              (cos: f32)
              (rho: f32)
              (i: i32)
              (halfsize: i32)
              (vct: [n]f32) : f32 =
          let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
          let ((lmin,xmin,ymin),(lplus,xplus,yplus)) = intersect_steep i ext ent halfsize
          let size = halfsize*2
          let pixmin = xmin+ymin*size
          let pixplus = xplus+yplus*size
          let min = if xmin >= 0 && xmin < size && ymin >=0 && ymin < size then (unsafe lmin*vct[pixmin]) else 0.0f32
          let plus = if  xplus >= 0 && xplus < size && yplus >=0 && yplus < size then (unsafe lplus*vct[pixplus]) else 0.0f32
          in (min+plus)

     -- calculate one value in the forward projection vector
     let forward_projection_value (sin: f32) (cos: f32) (rho: f32) (halfsize: i32) (img: []f32): f32 =
          reduce (+) 0.0f32 <| map(\i -> calculate_product sin cos rho i halfsize img)((-halfsize)...(halfsize-1))

     let fp [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32): []f32 =
          flatten <| map(\(cos, sin, _)->
               let k = sin/cos
               in map(\r ->
                    let rho = rhozero + r32(r)*deltarho
                    in forward_projection_value sin cos rho halfsize img
                    ) (iota numrhos)
           ) lines

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

     -- only works when lines have slope > 1. To use for all lines use preprocess to transpose lines and image
     let bp [p] [l] (lines: [l](f32, f32, f32))
               (rhozero: f32)
               (deltarho: f32)
               (rhosprpixel: i32)
               (numrhos: i32)
               (halfsize: i32)
               (projections: [p]f32) : []f32 =
          let fact = f32.sqrt(2.0f32)/2.0f32
          in flatten <| map(\irow ->
                         map(\icolumn ->
                              let xmin = r32(icolumn)
                              let ymin = r32(irow)
                              in reduce (+) 0.0f32 <| map(\ln ->
                                   let (cost, sint, lbase) = unsafe lines[ln]
                                   let tant = sint/cost
                                   let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                                   let rho = cost*p.1+sint*p.2
                                   let s = f32.ceil((rho-rhozero)/deltarho)
                                   let xbase = ymin*tant
                                   in reduce (+) 0.0f32 <| map(\i ->
                                        let sprime = s+(r32(i))
                                        let r = sprime*deltarho+rhozero
                                        let x_bot = (r/cost)-xbase
                                        let x_top = x_bot-tant
                                        let maxx = f32.max x_bot x_top
                                        let minx = f32.min x_bot x_top
                                        let l = (intersect_fact maxx minx xmin (xmin+1.0))*(lbase)
                                        let projectionidx = ln*numrhos+(t32(sprime))
                                        in l*(unsafe projections[projectionidx])
                                   )(iota rhosprpixel)
                              ) (iota l)
                         )((-halfsize)...(halfsize-1))
                    )((-halfsize)...(halfsize-1))
}

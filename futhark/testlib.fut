module testlib = {
  let is_flat (cos: f32) (sin: f32): bool =
    f32.abs(sin) >= f32.abs(cos)

  let preprocess [a](angles: [a]f32): ([](f32, f32, f32), [](f32, f32, f32)) =
    let cossin = map(\angle ->
      let cos= f32.cos(angle)
      let sin = f32.sin(angle)
      let lcot = f32.sqrt(1.0+(cos/sin)**2.0f32)
      let ltan = f32.sqrt(1.0+(sin/cos)**2.0f32)
      in (cos, sin, lcot, ltan)
    ) angles
    let parts = partition(\(c,s,_,_) -> is_flat c s ) cossin
    in (map (\(cos, sin, lcot, _)-> (cos, sin, lcot)) parts.1, map(\(cos, sin, _, ltan)-> (cos, sin, ltan)) parts.2)

  let forwardprojection_steep [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    let fhalfsize = r32(halfsize)
    let size = halfsize*2
    in flatten <| map (\(cos, sin, lbase) ->
      map (\r ->
        let rho = rhozero + r32(r)*deltarho
        let ent = ((rho-(-fhalfsize)*sin)/cos, (-fhalfsize))
        let ext = ((rho-fhalfsize*sin)/cos, fhalfsize)
        let k = (ext.1 - ent.1)/(ext.2 - ent.2)
        let kbase = ent.1 + fhalfsize

        in reduce (+) 0.0f32 <| map(\i ->
          let ih = i+halfsize

          let xmin = k*(r32(i) - ent.2) + kbase
          let xplus = k*(r32(i) + 1 - ent.2) + kbase

          let Xpixmin = t32(f32.floor(xmin))
          let Xpixplus = t32(f32.floor(xplus))

          let Xpixmax = r32(i32.max Xpixmin Xpixplus)
          let xdiff = xplus - xmin

          let bounds = ih >= 0 && ih < size
          let eq = Xpixmin == Xpixplus
          let bmin = bounds && Xpixmin >= 0 && Xpixmin < size
          let bplus = (!eq) && bounds && Xpixplus >= 0 && Xpixplus < size

          let lxminfac = ((Xpixmax - xmin)/xdiff)
          let lxmin = if eq then lbase else lxminfac*lbase
          let lxplus = ((xplus - Xpixmax)/xdiff)*lbase

          let pixmin = Xpixmin+ih*size
          let pixplus = Xpixplus+ih*size

          let min = if bmin then (unsafe lxmin*img[pixmin]) else 0.0f32
          let plus = if bplus then (unsafe lxplus*img[pixplus]) else 0.0f32

          in (min+plus)
        ) ((-halfsize)...(halfsize-1))
      ) (iota numrhos)
    ) lines

  let forwardprojection_flat [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    let fhalfsize = r32(halfsize)
    let size = halfsize*2
    in flatten <| map (\(cos, sin, lbase) ->
      map (\r ->
        let rho = rhozero + r32(r)*deltarho
        let ent = ((-fhalfsize), (rho-(-fhalfsize)*cos)/sin)
        let ext = (fhalfsize, (rho-fhalfsize*cos)/sin)
        let k = (ext.2 - ent.2)/(ext.1 - ent.1)
        let kbase = ent.2 + fhalfsize

        in reduce (+) 0.0f32 <| map(\i ->
          let ih = i+halfsize

          let ymin = k*(r32(i) - ent.1) + kbase
          let yplus = k*(r32(i) + 1 - ent.1) + kbase

          let Ypixmin = t32(f32.floor(ymin))
          let Ypixplus = t32(f32.floor(yplus))
          let Ypixmax = r32(i32.max Ypixmin Ypixplus)
          let ydiff = yplus - ymin

          -- bools
          let bounds = ih >= 0 && ih < size
          let eq = Ypixmin == Ypixplus
          let bmin = bounds && Ypixmin >=0 && Ypixmin < size
          let bplus = (!eq) && bounds && Ypixplus >=0 && Ypixplus < size

          let lyminfac = ((Ypixmax - ymin)/ydiff)
          let lymin = if eq then lbase else lyminfac*lbase
          let lyplus = ((yplus - Ypixmax)/ydiff)*lbase

          let pixmin = ih+Ypixmin*size
          let pixplus = ih+Ypixplus*size

          let min = if bmin then (unsafe lymin*img[pixmin]) else 0.0f32
          let plus = if bplus then (unsafe lyplus*img[pixplus]) else 0.0f32

          in (min+plus)
          )((-halfsize)...(halfsize-1))
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

  let bp_steep [p] [l] (lines: [l](f32, f32, f32))
    (offset:i32)
    (rhozero: f32)
    (deltarho: f32)
    (rhosprpixel: i32)
    (numrhos: i32)
    (halfsize: i32)
    (projections: [p]f32) : []f32 =
    let fact = f32.sqrt(2.0f32)/2.0f32
    in flatten (map(\irow ->
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
            let projectionidx = (ln+offset)*numrhos+(t32(sprime))
            in l*(unsafe projections[projectionidx])
          )(iota rhosprpixel)
        ) (iota l)
      )((-halfsize)...(halfsize-1))
    )((-halfsize)...(halfsize-1)))

  let bp_flat [p][l] (lines: [l](f32, f32, f32))
    (offset:i32)
    (rhozero: f32)
    (deltarho: f32)
    (rhosprpixel: i32)
    (numrhos: i32)
    (halfsize: i32)
    (projections: [p]f32): []f32 =
    let fact = f32.sqrt(2.0f32)/2.0f32
      in flatten (map(\irow ->
        map(\icolumn ->
          let xmin = r32(icolumn)
          let ymin = r32(irow)
          in reduce (+) 0.0f32 <| map(\ln ->
            let (cost, sint, lbase) = unsafe lines[ln]
            let cott = cost/sint
            let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
            let rho = cost*p.1+sint*p.2
            let s = f32.ceil((rho-rhozero)/deltarho)
            let ybase = xmin*cott
            in reduce (+) 0.0f32 <| map(\i ->
              let sprime = s+(r32(i))
              let r = sprime*deltarho+rhozero
              let y_left = (r/sint)-ybase
              let y_right = y_left-cott
              let maxy = f32.max y_left y_right
              let miny = f32.min y_left y_right
              let l = (intersect_fact maxy miny ymin (ymin+1.0))*lbase
              let projectionidx = (ln+offset)*numrhos+(t32(sprime))
              in l*(unsafe projections[projectionidx])
            )(iota rhosprpixel)
          ) (iota l)
        )((-halfsize)...(halfsize-1))
      )((-halfsize)...(halfsize-1)))

}

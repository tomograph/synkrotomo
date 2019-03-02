let forwardprojection_flat [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, _) ->
  map (\r ->
    let rho = rhozero + r32(r)*deltarho
    let fpv = map (\i ->
      let ent = ((-fhalfsize), find_y (-fhalfsize) rho cos sin)
      let ext = (fhalfsize, find_y fhalfsize rho cos sin)

      let k = (ext.2 - ent.2)/(ext.1 - ent.1)
      let ymin = k*(r32(i) - ent.1) + ent.2 + fhalfsize
      let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + fhalfsize
      let Ypixmin = f32.floor(ymin)
      let Ypixplus = f32.floor(yplus)

      let Ypixmax = f32.max Ypixmin Ypixplus
      let ydiff = yplus - ymin

      let b = if f32.abs(Ypixmin - Ypixplus) < 0.4f32 then true else false
      let bmin = if Ypixmin >= (-0.4f32) && Ypixmin < (r32(size) + 0.4f32) then true else false
      let bplus = if (not b) && Ypixplus >= (-0.4f32) && Ypixplus < (r32(size) + 0.4f32) then true else false

      let yminfacttmp = (r32(Ypixmax) - ymin)/ydiff
      let yminfact = if b then 1 else xminfacttmp
      let yplusfact = (yplus - Ypixmax)/ydiff

      let lymin = yminfact*lbase
      let lyplus = yplusfact*lbase

      let pixminval = lymin*(unsafe img[(i+halfsize)+t32(Ypixmin)*size])
      let pixplusval = lyplus*(unsafe img[(i+halfsize)+t32(Ypixplus)*size])

      let min = if bmin then pixminval else 0.0f32
      let plus = if bplus then pixplusval else 0.0f32

      in (min+plus)
    ) ((-halfsize)...(halfsize-1))
    in (reduce (+) 0.0f32 fpv)
  ) (iota numrhos)
) lines

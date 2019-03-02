let forwardprojection_steep [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, _) ->
    map (\r ->
      let rho = (rhozero + r32(r)*deltarho)
      let fpv = map (\i ->
        let ent = (find_x (-fhalfsize) rho cos sin, (-fhalfsize))
        let ext = (find_x fhalfsize rho cos sin, fhalfsize)

        let k = (ext.1 - ent.1)/(ext.2 - ent.2)
        let xmin = k*(r32(i) - ent.2) + ent.1 + (fhalfsize)
        let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (fhalfsize)
        let Xpixmin = f32.floor(xmin)
        let Xpixplus = f32.floor(xplus)

        let Xpixmax = f32.max Xpixmin Xpixplus
        let xdiff = xplus - xmin

        let b = if f32.abs(Xpixmin - Xpixplus) < 0.4f32 then true else false
        let bmin = if Xpixmin >= (-0.4f32) && Xpixmin < (r32(size) + 0.4f32) then true else false
        let bplus = if (not b) && Xpixplus >= (-0.4f32) && Xpixplus < (r32(size) + 0.4f32) then true else false

        -- if both equal then l is lbase and we only want one l

        let xminfacttmp = (Xpixmax - xmin)/xdiff
        let xminfact = if b then 1 else xminfacttmp
        let xplusfact = (xplus - Xpixmax)/xdiff

        let lxmin = xminfact*lbase
        let lxplus = xplusfact*lbase

        let pixminval = lxmin*(unsafe img[t32(Xpixmin)+(i+halfsize)*size])
        let pixplusval = lxplus*(unsafe img[t32(Xpixplus)+(i+halfsize)*size])

        let min = if bmin then pixminval else 0.0f32
        let plus = if bplus then pixplusval else 0.0f32

        in (min+plus)
      ) ((-halfsize)...(halfsize-1))
      in (reduce (+) 0.0f32 fpv)
    ) (iota numrhos)
  ) lines












  let forwardprojection_steep [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
    let fhalfsize = r32(halfsize)
    in flatten <| map (\(cos, sin, lbase, _) ->
      map (\r ->
        let rho = (rhozero + r32(r)*deltarho)
        let fpv = map (\i ->
          let ent = (find_x (-1.0*(fhalfsize)) rho cos sin, (-1.0*(fhalfsize)))
          let ext = (find_x (fhalfsize) rho cos sin, (fhalfsize))

          -- let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
          let k = (ext.1 - ent.1)/(ext.2 - ent.2)
          let xmin = k*(r32(i) - ent.2) + ent.1 + (fhalfsize)
          let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (fhalfsize)
          let Xpixmin = f32.floor(xmin)
          let Xpixplus = f32.floor(xplus)

          let Xpixmax = f32.max Xpixmin Xpixplus
          let xdiff = xplus - xmin

          let b = if f32.abs(Xpixmin - Xpixplus) < 0.4f32 then true else false

          let xminfact = if b then 1 else (Xpixmax - xmin)/xdiff
          let xplusfact = if b then 0 else (xplus - Xpixmax)/xdiff

          let lxmin = xminfact*lbase
          let lxplus = xplusfact*lbase

          -- FIX this is just renaming
          let ((lmin,xmin),(lplus,xplus)) = ((lxmin, Xpixmin), (lxplus, Xpixplus))
          let size = halfsize*2
          let pixmin = lmin*(unsafe img[t32(xmin)+(i+halfsize)*size])
          let pixplus = lplus*(unsafe img[t32(xplus)+(i+halfsize)*size])
          let min = if t32(xmin) >= 0 && t32(xmin) < size then pixmin else 0.0f32
          let plus = if  t32(xplus) >= 0 && t32(xplus) < size then pixplus else 0.0f32
          in (min+plus)
        ) ((-halfsize)...(halfsize-1))
        in (reduce (+) 0.0f32 fpv)
      ) (iota numrhos)
    ) lines

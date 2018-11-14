import "line_lib"
open Lines
module Matrix =
{
     --- DOUBLE PARALLEL
     -- function which computes the weight of pixels in grid_column for ray with entry/exit p
     let calculate_weight(ent: point)
               (ext: point)
               (i: i32)
               (N: i32) : [](f32,i32) =
          let Nhalf = N/2
          -- handle all lines as slope < 1 reverse the others
          let slope = (ext.2 - ent.2)/(ext.1 - ent.1)
          let reverse = f32.abs(slope) > 1
          let gridentry = if reverse then (if slope < 0 then (-ent.2,ent.1) else (-ext.2,ext.1)) else ent
          let k = if reverse then (-1/slope) else slope

          --- calculate stuff
          let ymin = k*(r32(i) - gridentry.1) + gridentry.2 + r32(Nhalf)
          let yplus = k*(r32(i) + 1 - gridentry.1) + gridentry.2 + r32(Nhalf)
          let Ypixmin = t32(f32.floor(ymin))
          let Ypixplus = t32(f32.floor(yplus))
          let baselength = f32.sqrt(1+k*k)
          -- in [(baselength,Ypixmin),(baselength,Ypixplus)]
          let Ypixmax = i32.max Ypixmin Ypixplus
          let ydiff = yplus - ymin
          let yminfact = (r32(Ypixmax) - ymin)/ydiff
          let yplusfact = (yplus - r32(Ypixmax))/ydiff
          let lymin = yminfact*baselength
          let lyplus = yplusfact*baselength
          let iindex = i+Nhalf
          -- index calculated wrong for reversed lines i think
          let pixmin = if reverse then (N-iindex-1)*N+Ypixmin else iindex+Ypixmin*N
          let pixplus = if reverse then (N-iindex-1)*N+Ypixplus else iindex+Ypixplus*N
          --let pixnon = if -- find pixel that is not crossed by line
          let min = if (pixmin >= 0 && pixmin < N ** 2) then
               (if Ypixmin == Ypixplus then (baselength,pixmin) else (lymin,pixmin))
               else (-1f32,-1i32)
          let plus = if (pixplus >= 0 && pixplus < N ** 2) then
               (if Ypixmin == Ypixplus then (-1f32,-1i32) else (lyplus,pixplus))
               else (-1f32,-1i32)
          in [min,plus]

     -- assuming  gridsize even
     let weights_doublepar(angles: []f32) (rays: []f32) (gridsize: i32): [][](f32,i32) =
          let halfgridsize = gridsize/2
          let entryexitpoints =  convert2entryexit angles rays (r32(halfgridsize))
          in map(\(ent,ext) -> (flatten(map (\i ->
                    calculate_weight ent ext i gridsize
               )((-halfgridsize)...(halfgridsize-1))))) entryexitpoints

     -- integrated version, i.e no matrix storage
     let intersect_steep (rho: f32) (i: i32) (k: f32) (ent: point) (Nhalf: i32): ((f32,i32,i32),(f32,i32,i32)) =
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

     let intersect_flat (rho: f32) (i: i32) (k: f32) (ent: point) (Nhalf: i32): ((f32,i32,i32),(f32,i32,i32)) =
          let ymin = k*(r32(i) - ent.1) + ent.2 + (r32(Nhalf))
          let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + (r32(Nhalf))
          let Ypixmin = t32(f32.floor(ymin))
          let Ypixplus = t32(f32.floor(yplus))
          -- could be done for all rays of same angle at once
          let baselength = f32.sqrt(1+k*k)
          let Ypixmax = i32.max Ypixmin Ypixplus
          let ydiff = yplus - ymin
          -- if both equal then l is baselength and we only want one l
          let yminfact = if Ypixmin == Ypixplus then 1 else (r32(Ypixmax) - ymin)/ydiff
          let yplusfact = if Ypixmin == Ypixplus then 0 else (yplus - r32(Ypixmax))/ydiff
          let lymin = yminfact*baselength
          let lyplus = yplusfact*baselength
          let x = i+Nhalf
          -- return with x, y switched since image has been transposed
          in ((lymin, x, Ypixmin), (lyplus, x, Ypixplus))

     -- mareika from Hamburg says they have special implementation of cone beam FDK from astra for GPU as it can not handle large datasets
     let calculate_product [n](sin: f32)
               (cos: f32)
               (rho: f32)
               (i: i32)
               (halfsize: i32)
               (vct: [n]f32) : f32 =
          let (ent,ext) = entryexitPoint sin cos rho (r32(halfsize))
          -- could be done for all rays of same angle at once
          let vertical = (ext.1 - ent.1) == 0
          let k = (ext.2 - ent.2)/(ext.1 - ent.1)
          let flat = f32.abs(k) < 1
          let ((lmin,xmin,ymin),(lplus,xplus,yplus)) = if flat then intersect_flat rho i k ent halfsize else intersect_steep rho i k ent halfsize
          let size = halfsize*2
          let pixmin = xmin+ymin*size
          let pixplus = xplus+yplus*size
          --let pixnon = if -- find pixel that is not crossed by line
          let min = if vertical then unsafe vct[(t32(f32.floor(rho)))+(i+halfsize)*size] else (if pixmin >= 0 && pixmin < size**2 then (unsafe lmin*vct[pixmin]) else 0)
          let plus = if vertical then 0 else (if pixplus >= 0 && pixplus < size**2 then (unsafe lplus*vct[pixplus]) else 0)
          in (min+plus)

     -- loops are intechanged and no matrix values are saved
     -- in future only do half of the rhos by mirroring but concept needs more work.
     -- problem with copying of arrays causes memory issues. Don't copy stuff
     -- let projection_difference [a][r][n][p](angles: [a]f32) (rhos: [r]f32) (img: [n]f32) (projections: *[p]f32): [p]f32 =
     let projection_difference [a][r][n](angles: [a]f32) (rhos: [r]f32) (img: [n]f32) : []f32 =
          let halfsize = r/2
          --let transposedimg = transpose img
          in flatten(
               (map(\ang->
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    -- transpose image (rotate) so that when taking a row of the matrix its actually a column when need be
                    --let imrow = unsafe(if flat then (transposedimg[i+halfsize]) else img[i+halfsize])
                    --let proj = (unsafe projections[j*r+i+halfsize])-- secial case of r = n, then i+halfsize is also index for ray
                    in (map(\o -> reduce (+) 0 <| map(\i -> calculate_product sin cos o i halfsize img)((-halfsize)...(halfsize-1))) rhos)
               ) (angles)))
}

-- open Matrix
-- let main  (x1:f32)
--           (y1: f32)
--           (x2: f32)
--           (y2: f32)
--           (i: i32)
--           (N: i32): ([]f32, []i32) =
--           unzip(calculate_weight (x1,y1) (x2,y2) i N)

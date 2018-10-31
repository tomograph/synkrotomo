-- ==
-- input@../test
import "intersect_lib_map"
import "projection_lib"
open Intersections
open Projection
let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
    xs |> map unzip
         |> unzip

let forwardprojection (rays : []f32)
                               (angles : []f32)
                               (voxels : []f32) =
    let gridsize = t32(f32.sqrt(r32((length voxels))))
    let halfsize = r32(gridsize)/2
    let entrypoints = convert2entry angles rays halfsize
    let totalLen = (length entrypoints)
    let stepSize = 2
    -- let runLen = if (totalLen/stepSize) == 0 then 1 else (totalLen/stepSize)
    let runLen = (totalLen/stepSize)
    let testmat = replicate (gridsize*2-1) (-1f32, -1i32)
    let (testmat, _, _, _, _, _, _) =
        loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = ([testmat], 0, runLen, stepSize, gridsize, entrypoints, totalLen)
        while ( run < runLen ) do
            let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
            let part = map (\s -> unsafe (lengths gridsize (entrypoints[run*stepSize + s].2).1 (entrypoints[run*stepSize + s].2).2 entrypoints[run*stepSize + s].1 )) (iota step)
            in (output++part, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
    in (unzip_d2 (tail testmat))
                   -- let intersections = map (\rl ->
                   --   map (\s ->
                   --     (lengths gridsize entrypoints[rl*stepSize + s].2.1 entrypoints[rl*stepSize + s].2.2 entrypoints[rl*stepSize + s].1)
                   --   ) (iota stepSize)
                   -- ) (iota runLen)
                   -- let out = []
                   -- in unzip_d2( unsafe flatten (map (\rl ->
                   --   let css = if (stepSize * (rl+1)) > totalLen then (stepSize * (rl+1)) - totalLen else stepSize
                   --   in (map (\s ->
                   --     let index = (stepSize*rl + s)
                   --     in unsafe(lengths gridsize ((entrypoints[index].2).1) ((entrypoints[index].2).2)  (entrypoints[index].1) )
                   --   ) (unsafe(iota css)))
                   -- ) (iota runLen)))

                   -- in unzip_d2( flatten(map (\rl ->
                   --   map (\s ->
                   --     unsafe (lengths gridsize (entrypoints[rl*stepSize + s].2).1 (entrypoints[rl*stepSize + s].2).2 entrypoints[rl*stepSize + s].1)
                   --   ) (iota stepSize)
                   -- ) (iota runLen)))

let main  (rays: []f32)
          (angles: []f32)
          (pixels: []f32) =
          forwardprojection rays angles pixels

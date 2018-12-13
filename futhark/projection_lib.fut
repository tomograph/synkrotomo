import "matrix_lib"
open Matrix

module Projection = {

      -- not sparse, nested, matrix vector multiplication for backprojection, uses a padded version of the matrix
     --  let notSparseMatMult_back [num_rows] [num_cols]
     --                        (mat_vals : [num_rows][num_cols]f32)
     --                        (vect : []f32) : [num_rows]f32 =
     --      map (\row -> reduce (+) 0 <| map2 (*) row vect ) mat_vals
     --
     -- -- pads and transposes the matrix, nested, will perform better when tiling is improved in futhark
     -- let trans_map [m] [n]
     --                (matrix : [m][n](f32, i32)) (gridsize: i32): [][]f32 =
     --           let rs = gridsize*gridsize
     --           let padded = map (\row -> let (vals, inds) = (unzip row) in scatter (replicate rs 0.0f32) inds vals ) matrix
     --           in transpose padded

     -- let backprojection  [a][r][p](angles : [a]f32)
     --                     (rhos : [r]f32)
     --                     (projections : [p]f32)
     --                     (gridsize: i32)
     --                     (stepSize : i32) : []f32=
     --           let halfsize = r32(gridsize)/2
     --           let entryexitpoints =  convert2entryexit angles rhos halfsize
     --           let runLen = (p/stepSize)
     --           -- result array
     --           let backmat = replicate (gridsize*gridsize) 0.0f32
     --           -- stripmined, sequential outer loop, mapped inner
     --           let (backmat, _, _, _, _, _, _) =
     --               loop (output, run, runLen, stepSize, gridsize, entryexitpoints, p) = (backmat, 0, runLen, stepSize, gridsize, entryexitpoints, p)
     --               while ( run < runLen ) do
     --                   -- if the number of entrypoints doesn't line perfectly up with the stepsize
     --                   let step = if (run+1)*stepSize >= p then p - run*stepSize else stepSize
     --                   -- calc part of matrix, stepSize rows
     --                   let halfgridsize = gridsize/2
     --
     --                   let partmatrix = map(\(ent,ext) -> (flatten(map (\i ->
     --                             calculate_weight ent ext i gridsize
     --                        )((-halfgridsize)...(halfgridsize-1))))) (entryexitpoints[(run*stepSize) : (run*stepSize + step)])
     --
     --                   -- transpose
     --                   let transmat = trans_map partmatrix gridsize
     --                   in ((map2 (+) ((notSparseMatMult_back transmat projections[(run*stepSize) : (run*stepSize + step)])) output), run+1, runLen, stepSize, gridsize, entryexitpoints, p)
     --           in backmat

     -- in future only do half of the rhos by mirroring but concept needs more work.
     -- problem with copying of arrays causes memory issues. Don't copy stuff
     let forward_projection [a][r][n](angles: [a]f32) (rhos: [r]f32) (halfsize: i32) (img: [n]f32): []f32 =
          flatten(
               (map(\i->
                    let ang = unsafe angles[i]
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    in (map(\r -> forward_projection_value sin cos r halfsize img) rhos)
               ) (iota a)))

     let getprojectionindex (angleindex: i32) (rhovalue: f32) (deltarho: f32) (rhozero: f32) (numrhos: i32): i32 =
          angleindex*numrhos+t32((rhovalue-rhozero)/deltarho)

     let back_projection [a][p] (angles: [a]f32) (rhozero: f32) (deltarho: f32) (size: i32) (projections: [p]f32): []f32=
          let rhosforpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          in map(\pix ->
               let pixcenter = pixelcenter pix size
               in reduce (+) 0 <| (flatten(map(\i ->
                    let ang = unsafe angles[i]
                    let sin = f32.sin(ang)
                    let cos = f32.cos(ang)
                    let minrho = rhomin cos sin pixcenter rhozero deltarho
                    let rhos = getrhos minrho deltarho rhosforpixel
                    in (map(\rho->
                              let l = intersectiondistance sin cos rho pixcenter
                              let projectionidx = getprojectionindex i rho deltarho rhozero (p/a)
                              in l*(unsafe projections[projectionidx])
                         ) rhos)
               ) (iota a)))
          )(iota (size**2))


}

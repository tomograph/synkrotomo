-- compiled input {
--  -1.0f32
--  -1.0f32
--  0.0f32
--  0.0f32
--  0.0f32
--  2.0f32
-- 2.0f32
-- 0.0f32
-- }
-- output { 1.41421356237 }

import "vspace"
import "vector"

module vec3 = mk_vspace_3d f32

type face  = {n: vec3.vector, d: f32}
type voxel = {xy: face, yz: face, xz: face}
let unvec3 ({x,y,z}: vec3.vector) = (x,y,z)
--identifies voxel by normals and distances, assumes isotropic unit one sides
let get_voxel (lower_left: vec3.vector) : voxel =
     let n1 = {x=(-1.0f32),y=0.0f32,z=0.0f32}
     let n2 = {x=0.0f32,y=(-1.0f32),z=0.0f32}
     let n3 = {x=0.0f32,y=0.0f32,z=(-1.0f32)}
     let d1 = lower_left.x
     let d2 = lower_left.y
     let d3 = lower_left.z
     let xyface = {n=n3,d=d3}
     let yzface = {n=n1,d=d1}
     let xzface = {n=n2,d=d2}
     in {xy=xyface, yz=yzface, xz=xzface}

let scalar (dist: f32) (source: vec3.vector) (normal: vec3.vector) (ray_dir: vec3.vector) =
     let divisor = vec3.dot ray_dir normal
     let denominator = dist - (vec3.dot source normal)
     in (denominator,divisor)

let maxfrac (v1: (f32,f32)) (v2: (f32,f32)): (f32,f32) =
     if v1.1/v1.2 > v2.1/v2.2 then v1 else v2

let minfrac (v1: (f32,f32)) (v2: (f32,f32)): (f32,f32) =
     if v1.1/v1.2 < v2.1/v2.2 then v1 else v2

let min_pos (v1: (f32,f32)) (v2: (f32,f32)) =
     if v1.2 > 0.0f32 && v2.2 > 0.0f32 then minfrac v1 v2 else if v1.2 > 0.0f32 then v1 else if v2.2 > 0.0f32 then v2 else (f32.nan,f32.nan)

let max_neg (v1: (f32,f32)) (v2: (f32,f32)) =
     if v1.2 < 0.0f32 && v2.2 < 0.0f32 then maxfrac v1 v2 else if v1.2 < 0.0f32 then v1 else if v2.2 < 0.0f32 then v2 else (0.0f32,0.0f32)

let min_pos_six (v1: (f32,f32)) (v2: (f32,f32)) (v3: (f32,f32)) (v4: (f32,f32)) (v5: (f32,f32)) (v6: (f32,f32)) =
     let w1 = min_pos v1 v2
     let w2 = min_pos v3 v4
     let w3 = min_pos v5 v6
     let u1 = min_pos w1 w2
     let minimum = min_pos u1 w3
     in minimum.1/minimum.2

let max_neg_six (v1: (f32,f32)) (v2: (f32,f32)) (v3: (f32,f32)) (v4: (f32,f32)) (v5: (f32,f32)) (v6: (f32,f32)) =
     let w1 = max_neg v1 v2
     let w2 = max_neg v3 v4
     let w3 = max_neg v5 v6
     let u1 = max_neg w1 w2
     let maximum = max_neg u1 w3
     in maximum.1/maximum.2

let get_extreme_scalars (v: voxel) (ray_dir: vec3.vector) (source: vec3.vector) =
     let xy_bot = scalar v.xy.d source v.xy.n ray_dir
     let xy_top = scalar (v.xy.d+1) source (vec3.scale (-1.0f32) v.xy.n) ray_dir
     let yz_left = scalar v.yz.d source v.yz.n ray_dir
     let yz_right = scalar (v.yz.d+1) source (vec3.scale (-1.0f32) v.yz.n) ray_dir
     let xz_front = scalar v.xz.d source v.xz.n ray_dir
     let xz_behind = scalar (v.xz.d+1) source (vec3.scale (-1.0f32) v.xz.n) ray_dir
     -- in ((xy_bot.1/xy_bot.2), (xy_top.1/xy_top.2), (yz_left.1/yz_left.2), (yz_right.1/yz_right.2), (xz_front.1/xz_front.2), (xz_behind.1/xz_behind.2))
     let mp = min_pos_six xy_bot xy_top yz_left yz_right xz_front xz_behind
     let mn = max_neg_six xy_bot xy_top yz_left yz_right xz_front xz_behind
     in (mn,mp)
--
let box_ray_intersect (ray_source: vec3.vector) (ray_dest: vec3.vector) (lower_left: vec3.vector) : f32 =
     let ray_direction =  ray_dest vec3.- ray_source
     let v = get_voxel(lower_left)
     let (tin,tout) = get_extreme_scalars v ray_direction ray_source
     let t = tout-tin
     let tr = vec3.scale t ray_direction
     in vec3.norm tr

let main (x1: f32) (y1: f32) (z1: f32) (x2: f32) (y2: f32) (z2: f32) (x3: f32) (y3: f32) (z3: f32) =
     box_ray_intersect {x=x1,y=y1,z=z1} {x=x3,y=y3,z=z3} {x=x2,y=y2,z=z2}

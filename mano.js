import * as THREE from 'three';

/* Uncomment below to run mano_test.js in node.js*/
import * as tf from '@tensorflow/tfjs';

export {
    MANO,

    Rodrigues_forward,
    lrotmin,
    posemap_axisang,
    batch_rodrigues,
    quat2mat,
    subtract_flat_id,
    with_zeros,

    _bmm,
};


// rt: rotation vector, tensor (3,)
function Rodrigues_forward(vec) {
    const norm = tf.norm(vec, 2);
    const k = vec.div(norm).dataSync();
    const theta = norm;
    let kx, ky, kz;
    kx = k[0];
    ky = k[1];
    kz = k[2];
    const K = tf.tensor([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    const K2 = tf.matMul(K, K);
    const s = tf.sin(theta);
    const c = tf.cos(theta);
    const R = tf.eye(3).add(K.mul(s)).sub(K2.mul(c.sub(1)));
    return R;
}

// p: tensor (48,)
function lrotmin(p) {
    let p2 = p.reshape([-1, 3]);
    // p2 = p2.slice([1, 0], [1, 3]);
    let Rs = []
    for (let i = 1; i < p2.shape[0]; i++) {
        let R = Rodrigues_forward(p2.slice([i, 0], [1, 3])).sub(tf.eye(3)).flatten()
        Rs.push(R);
    }
    Rs = tf.stack(Rs).flatten();
    return Rs;
}

function posemap(s) {
    if (s == 'lrotmin')
        return lrotmin;
}

// pose_vec: tensor (B, 48)
// => [pose_maps: tensor (B, 144), rot_mats: tensor (B, 144) ]
function posemap_axisang(pose_vec) {
    const rot_nb = pose_vec.shape[1] / 3;
    const pose_vec_reshaped = pose_vec.reshape([-1, 3]);
    let rot_mats = batch_rodrigues(pose_vec_reshaped);
    rot_mats = rot_mats.reshape([-1, rot_nb*9]);
    const pose_maps = subtract_flat_id(rot_mats);
    return [pose_maps, rot_mats];
}

// tensor (B, 3, 4)
// => (B, 4, 4)
function with_zeros(tensor) {
    const bs = tensor.shape[0];
    const padding = tf.tensor([0.0, 0.0, 0.0, 1.0]);
    const cat_res = tf.concat([
        tensor, padding.reshape([1, 1, 4]).tile([bs, 1, 1])
    ], 1);
    return cat_res;
}

// axisang: tensor (16, 3)
// => (16, 9)
function batch_rodrigues(axisang) {
    const axisang_norm = tf.norm(axisang.add(1e-8), 2, 1)
    let angle = tf.expandDims(axisang_norm, -1);
    const axisang_normalized = tf.div(axisang, angle);
    angle = angle.mul(0.5);
    const v_cos = tf.cos(angle);
    const v_sin = tf.sin(angle);
    const quat = tf.concat([v_cos, tf.mul(axisang_normalized, v_sin)], 1);
    let rot_mat = quat2mat(quat);
    rot_mat = rot_mat.reshape([-1, 9]);
    return rot_mat;
}

// quat: tensor (16, 4)
// => (16, 3, 3)
function quat2mat(quat) {
    const norm_quat = tf.div(quat, tf.norm(quat, 2, 1, true));
    const bs = norm_quat.shape[0];
    const [w, x, y, z] = tf.unstack(norm_quat, 1);
    const [w2, x2, y2, z2] = [w.pow(2), x.pow(2), y.pow(2), z.pow(2)];
    const [wx, wy, wz] = [w.mul(x), w.mul(y), w.mul(z)];
    const [xy, xz, yz] = [x.mul(y), x.mul(z), y.mul(z)];

    const rotMat = tf.stack([
        w2.add(x2).sub(y2).sub(z2), xy.mul(2).sub(wz.mul(2)), wy.mul(2).add(xz.mul(2)), wz.mul(2).add(xy.mul(2)),
        w2.sub(x2).add(y2).sub(z2), yz.mul(2).sub(wx.mul(2)), xz.mul(2).sub(wy.mul(2)), wx.mul(2).add(yz.mul(2)),
        w2.sub(x2).sub(y2).add(z2),
    ], 1).reshape([-1, 3, 3]);
    return rotMat;
}

// rot_mats: tensor (B, 144)
// => (B, 144)
function subtract_flat_id(rot_mats) {
    const bs = rot_mats.shape[0];
    const rot_nb = rot_mats.shape[1] / 9;
    const id_flat = tf.tile(
        tf.eye(3, undefined, undefined, rot_mats.dtype).reshape([1, 9]),
        [bs, rot_nb])
    return tf.sub(rot_mats, id_flat);
}

// tensor1: (B, a, b)
// tensorb: (B, b, c)
// => (B, a, c)
function _bmm(tensor1, tensor2) {
    return tf.einsum('bij,bjk->bik', tensor1, tensor2);
}

class MANO {
    constructor(
        mano_object,
        center_idx=null,
        flat_hand_mean=true,
        ncomps=6,
        side='right',
        use_pca=true,
        root_rot_mode='axisang',
        joint_rot_mode='axisang',
        )
    {
        this.center_idx = center_idx;
        this.flat_hand_mean = flat_hand_mean;
        this.ncomps = ncomps;
        this.side = side;
        this.use_pca = use_pca;
        this.ncomps = use_pca ? ncomps : 45;
        this.root_rot_mode = root_rot_mode;
        this.joint_rot_mode = joint_rot_mode;
        if (root_rot_mode == 'axisang') {
            this.rot = 3
        } else {
            throw new Error("Not implemented");
        }

        // if (side == 'right') {
        //     this.mano_path = `${mano_root}/MANO_RIGHT.json`;
        // } else if (side == 'left') {
        //     this.mano_path = `${mano_root}/MANO_LEFT.json`;
        // }

        const dd = this.ready_argument(mano_object);
        this.betas = tf.expandDims(dd.betas, 0);
        this.shapedirs = dd.shapedirs;
        this.posedirs = dd.posedirs;
        this.v_template = tf.expandDims(dd.v_template, 0);
        this.J_regressor = dd.J_regressor;
        this.weights = dd.weights;
        this.faces = tf.cast(dd.f, 'int32');

        const hands_components = dd.hands_components;
        const hands_mean = tf.clone(
            flat_hand_mean ? tf.zeros([hands_components.shape[1]]) : dd.hand_mean);
        if (this.use_pca || (this.joint_rot_mode == 'axisang')) {
            this.hands_mean = hands_mean;
            this.comps = hands_components;
            this.select_comps = hands_components.slice([0], [ncomps]);
        } else {
            throw Error;
        }

        this.kintree_table = dd.kintree_table;

        // Three.js fields
        const geometry = new THREE.BufferGeometry();
        geometry.setIndex(Array.from(this.faces.dataSync()));
        const colors = [];
        for (let i = 0; i < 778; i++) {
            const r = i / 28 / 28;
            const g = (i % 28) / 28;
            colors.push(r, g, 0.5);
        }
        // Set init vertices
        geometry.setAttribute('position',
            new THREE.Float32BufferAttribute(this.v_template.dataSync(), 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        const material = new THREE.MeshPhongMaterial( {
            side: THREE.DoubleSide,
            vertexColors: true
        } );
        this.mesh = new THREE.Mesh(geometry, material);
        // const geo = new THREE.EdgesGeometry(this.mesh.geometry); // or WireframeGeometry
        // const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
        // const wireframe = new THREE.LineSegments(geo, mat);
        // this.mesh.add(wireframe);
    }

    ready_argument(data, posekey4vposed = 'pose') {
        let dd = {};
        for (const k in data) {
            const v = data[k];
            if ((k == 'bs_type') || (k == 'bs_style'))
                dd[k] = v;
            else
                dd[k] = tf.tensor(v);
        }

        const want_shapemodel = 'shapedirs' in dd;
        const nposeparms = dd['kintree_table'].shape[1] * 3;

        if (!('trans' in dd))
            dd.trans = tf.zeros([3]);
        if (!('pose' in dd))
            dd.pose = tf.zeros([nposeparms]);
        if (('shapedirs' in dd) && !('betas' in dd)) {
            dd.betas = tf.zeros([dd.shapedirs.shape.at(-1)]);
        }

        if (want_shapemodel) {
            dd.v_shaped = dd.shapedirs.reshape([-1, 10]).dot(dd.betas).reshape([-1, 3])
                .add(dd.v_template);
            const v_shaped = dd.v_shaped;
            const [v_x, v_y, v_z] = tf.unstack(v_shaped, 1); // (778, 3) => (778,) * 3
            const J_tmpx = dd.J_regressor.dot(v_x);
            const J_tmpy = dd.J_regressor.dot(v_y);
            const J_tmpz = dd.J_regressor.dot(v_z);
            dd.J = tf.stack([J_tmpx, J_tmpy, J_tmpz]).transpose();
            const pose_map_res = posemap(dd.bs_type)(dd[posekey4vposed]);
            dd.v_posed = v_shaped.add(
                dd.posedirs.reshape([-1, 135]).dot(pose_map_res).reshape([-1, 3]));
        } else {
            throw new Error("Not implemented");
        }

        return dd;
    }

    /*
     * poses: tensor (B, 48)
     * betas: tensor (B, 10) if provided, uses given shape parameters for hand shape
     *      else centers on root joint (9th joint)
     * trans: tensor (B, 3) if provided, applies trans to joints and vertices
     * root_palm: bool, return palm as hand root instead of wrist
    */
    forward(poses, betas, trans, root_palm=false) {
        const V = 778;
        const bsize = poses.shape[0];
        const hand_pose_coeffs = poses.slice([0, this.rot], [bsize, this.ncomps]);

        let pose_map, rot_map;
        let root_rot;
        if (this.use_pca || (this.joint_rot_mode == 'axisang')) {
            let full_hand_pose;
            if (this.use_pca) {
                full_hand_pose = hand_pose_coeffs.dot(this.select_comps);
            } else {
                full_hand_pose = hand_pose_coeffs;
            }
            const full_pose = tf.concat(
                [
                    poses.slice([0, 0], [bsize, this.rot]),
                    full_hand_pose.add(this.hands_mean)
                ], 1);

            if (this.root_rot_mode == 'axisang') {
                [pose_map, rot_map] = posemap_axisang(full_pose);
                root_rot = rot_map.slice([0, 0], [bsize, 9]).reshape([bsize, 3, 3]);
                rot_map = rot_map.slice([0, 9], [bsize, rot_map.shape[1] - 9]);
                pose_map = pose_map.slice([0, 9], [bsize, pose_map.shape[1] - 9]);
            } else {
                throw Error;
            }
        } else {
            throw Error;
        }

        // Full axis angle repre with root joint
        let v_shaped, j;
        if ((typeof(betas) == 'undefined') || (betas.size == 1)) {
            // this.shapedirs: (778, 3, 10)
            // this.betas: (B, 10) => (10, B)
            // (778, 3, 10) x (10, B) = (778, 3, B) => (B, 778, 3)
            v_shaped = tf.dot(
                this.shapedirs.reshape([V*3, 10]),
                this.betas.transpose([1, 0]))
                .reshape([V, 3, bsize]).transpose([2, 0, 1]).add(this.v_template);

            // this.J_regressor: (16, 778)
            // v_shaped: (B, 778, 3) => (778, B*3)
            // (16, 778) x (B, 778, 3) = (16, B*3) => (16, B, 3) => (B, 16, 3)
            j = tf.dot(
                this.J_regressor,
                v_shaped.transpose([1, 0, 2]).reshape([V, bsize*3])
                ).reshape([-1, bsize, 3]).transpose([1, 0, 2]);
        } else {
            throw Error;
        }

        const v_posed = v_shaped.add(
            tf.dot(
                this.posedirs.reshape([V*3, -1]), // (V*3, 135)
                pose_map.transpose([1, 0]) // (B, 135) => (135, B)
                ) // = (V*3, B)
            .reshape([778, 3, bsize]) // => (V, 3, B)
            .transpose([2, 0, 1]) // => (B, V, 3)
        ); // (B, 778, 3)


        // Global rigid transformation

        const root_j = j.slice([0, 0, 0], [bsize, 1, 3]).reshape([bsize, 3, 1]); // (B, 3, 1)
        const root_trans = with_zeros(tf.concat([root_rot, root_j], 2)); // (B, 4, 4)

        const all_rots = rot_map.reshape([bsize, 15, 3, 3]); // (B, 15, 3, 3)
        const lev1_idxs = [1, 4, 7, 10, 13];
        const lev2_idxs = [2, 5, 8, 11, 14];
        const lev3_idxs = [3, 6, 9, 12, 15];
        const lev1_rots = tf.gather(all_rots, lev1_idxs.map(x=>x-1), 1); // (1, 5, 3, 3)
        const lev2_rots = tf.gather(all_rots, lev2_idxs.map(x=>x-1), 1);
        const lev3_rots = tf.gather(all_rots, lev3_idxs.map(x=>x-1), 1);
        const lev1_j = tf.gather(j, lev1_idxs, 1); // (1, 5, 3)
        const lev2_j = tf.gather(j, lev2_idxs, 1);
        const lev3_j = tf.gather(j, lev3_idxs, 1);

        // From base to tips
        // Get lev1 results
        let all_transforms = [root_trans.expandDims(0)];
        const lev1_j_rel = lev1_j.sub(root_j.transpose([0, 2, 1])); // (1, 5, 3)
        const lev1_rel_transform_flt = with_zeros(
            tf.concat([lev1_rots, lev1_j_rel.expandDims(3)], 3)
            .reshape([-1, 3, 4]));
        const root_trans_flt = root_trans.expandDims(1).tile([1, 5, 1, 1,])
            .reshape([root_trans.shape[0]*5, 4, 4]);
        const lev1_flt = _bmm(root_trans_flt, lev1_rel_transform_flt);
        all_transforms.push(lev1_flt.reshape([bsize, 5, 4, 4]));

        // Get lev2 results
        const lev2_j_rel = lev2_j.sub(lev1_j); // (1, 5, 3)
        const lev2_rel_transform_flt = with_zeros(
            tf.concat([lev2_rots, lev2_j_rel.expandDims(3)], 3)
            .reshape([-1, 3, 4]));
        const lev2_flt = _bmm(lev1_flt, lev2_rel_transform_flt);
        all_transforms.push(lev2_flt.reshape([bsize, 5, 4, 4]));

        // Get lev2 results
        const lev3_j_rel = lev3_j.sub(lev2_j); // (1, 5, 3)
        const lev3_rel_transform_flt = with_zeros(
            tf.concat([lev3_rots, lev3_j_rel.expandDims(3)], 3)
            .reshape([-1, 3, 4]));
        const lev3_flt = _bmm(lev2_flt, lev3_rel_transform_flt);
        all_transforms.push(lev3_flt.reshape([bsize, 5, 4, 4]));

        const reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15];
        const results = tf.gather(tf.concat(all_transforms, 1), reorder_idxs, 1)
        const results_global = results;

        let _d1, _d2, _d3, _d4;
        const joint_js = tf.concat([j, tf.zeros([j.shape[0], 16, 1])], 2);
        _d1 = results.shape[0], _d2 = results.shape[1];
        const tmp2 = _bmm(
            results.reshape([_d1*_d2, 4, 4]), joint_js.reshape([_d1*_d2, 4, 1])
            ).reshape([_d1, _d2, 4, 1]);
        const _op2 = tf.concat([tf.zeros([_d1, _d2, 4, 3]), tmp2], 3);
        const results2 = results.sub(_op2).transpose([0, 2, 3, 1]);

        [_d1, _d2, _d3, _d4] = results2.shape;
        const T = tf.dot(
            results2.reshape([_d1*_d2*_d3, _d4]), this.weights.transpose([1, 0]))
            .reshape([_d1, _d2, _d3, 778]);  //  (B, 4, 4, 778)

        const rest_shape_h = tf.concat([
            v_posed.transpose([0, 2, 1]),
            tf.ones([bsize, 1, v_posed.shape[1]], T.dtype)], 1); // (B, 4, 778)

        let verts = T.mul(rest_shape_h.expandDims(1)).sum(2).transpose([0, 2, 1]);
        verts = verts.slice([0, 0, 0], [bsize, verts.shape[1], 3]);
        let jtr = results_global.slice([0, 0, 0, 3],
            [bsize, results_global.shape[1], 3, 1]).squeeze(3);
        let tips;
        if (this.side == 'right') {
            tips = tf.gather(verts, [745, 317, 444, 556, 673], 1);
        } else {
            tips = tf.gather(verts, [745, 317, 445, 556, 673], 1);
        }
        if (root_palm) {
            throw Error;
            const palm = tf.add(
                tf.gather(verts, 95, 1),
                tf.gather(verts, 22, 1)).expandDims(1).div(2);
            jtr = tf.concat([palm, jtr.slice([0, 1], [bsize, jtr.shape[1]-1])], 1);
        }
        jtr = tf.concat([jtr, tips], 1);
        jtr = tf.gather(jtr,
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
            1);

        if ((typeof(trans) == 'undefined') || (trans.norm().dataSync()[0] == 0)) {
            if (this.center_idx != null) {
                const center_joint = jtr.gather(this.center_idx, 1).expandDims(1);
                jtr = jtr.sub(center_joint);
                verts = verts.sub(center_joint);
            }
        } else {
            throw Error;
        }

        const scale = 1000.0;
        verts = verts.mul(scale);
        jtr = jtr.mul(scale);
        return [verts, jtr];
    }

    // three.js wrapper of this.forward()
    forward_mesh(poses, betas, trans, root_palm=false) {
        const [verts_th, jtr_th] = this.forward(poses, betas, trans, root_palm);
        const verts = verts_th.dataSync();

        for (let i = 0; i < verts_th.shape[0]; i++) {
            this.mesh.geometry.attributes.position.setXYZ(
                i, verts[3*i], verts[3*i+1], verts[3*i+2]);
        }
        this.mesh.geometry.attributes.position.needsUpdate = true;
        this.mesh.geometry.computeVertexNormals();

        // const scale = 100.0;
        // this.mesh.geometry.scale(scale, scale, scale);
        return this.mesh;
    }

    set_mesh(mesh, poses, betas, trans, root_palm=false) {
        const [verts_th, jtr_th] = this.forward(poses, betas, trans, root_palm);
        const verts = verts_th.dataSync();

        for (let i = 0; i < verts_th.shape[0]; i++) {
            mesh.geometry.attributes.position.setXYZ(
                i, verts[3*i], verts[3*i+1], verts[3*i+2]);
        }
        mesh.geometry.attributes.position.needsUpdate = true;
        console.log(verts_th.sum().dataSync());
        mesh.geometry.computeVertexNormals();
    }
}

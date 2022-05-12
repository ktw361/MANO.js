import * as tf from '@tensorflow/tfjs';
// import * as THREE from 'three';

export { Rodrigues_forward };


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
    Rs = tf.stack(Rs).flatten(); // TODO: ravel?
    return Rs;
}

function posemap(s) {
    if (s == 'lrotmin')
        return lrotmin;
}

function posemap_axisang(pose_vec) {
    const rot_nb = pose_vec.shape[0] / 3;
    const pose_vec_reshaped = pose_vec.reshape([-1, 3]);
    let rot_mats = batch_rodrigues(pose_vec_reshaped);
    rot_mats = rot_mats.reshape([-1, rot_nb*9]);
    const pose_maps = subtract_flat_id(rot_mats);
    return pose_maps, rot_mats;
}

// axisang: tensor (16, 3)
function batch_rodrigues(axisang) {
    const axisang_norm = tf.norm(axisang.add(1e-8), ord=2, axis=1)
    const angle = tf.expandDims(axisang_norm, -1);
    const axisang_normalized = tf.div(axisang, angle);
    angle = angle.mul(0.5);
    const v_cos = tf.cos(angle);
    const v_sin = tf.sin(angle);
    const quat = tf.concat([v_cos, tf.mul(axisang_normalized, v_sin)], axis=1);
    const rot_mat = quat2mat(quat);
    rot_mat = rot_mat.reshape([-1, 9]);
    return rot_mat;
}
function batch_rodrigues_test() {

}

// quat: tensor (16, 4)
function quat2mat(quat) {
    const norm_quat = tf.div(quat, tf.norm(quat, ord=2, axis=1, keepdims=True));
    const bs = norm_quat.shape[0];
    const w = norm_quat.slice([0, 0], [bs, 1]).flatten(),
          x = norm_quat.slice([0, 1], [bs, 1]).flatten(),
          y = norm_quat.slice([0, 2], [bs, 1]).flatten(),
          z = norm_quat.slice([0, 3], [bs, 1]).flatten();
    const w2 = w.pow(2),
          x2 = w.pow(2),
          y2 = y.pow(2),
          z2 = z.pow(2);
    const wx = tf.mul(w, x);
    const wy = tf.mul(w, y);
    const wz = tf.mul(w, z);
    const xy = tf.mul(x, y);
    const xz = tf.mul(x, z);
    const yz = tf.mul(y, z);

    const rotMat = tf.stack([
        w2.add(x2).sub(y2).sub(z2), xy.mul(2).sub(wz.mul(2)), wy.mul(2).add(xz.mul(2)), wz.mul(2).add(xy.mul(2)),
        w2.sub(x2).add(y2).sub(z2), yz.mul(2).sub(wx.mul(2)), xz.mul(2).sub(wy.mul(2)), wx.mul(2).add(yz.mul(2)),
        w2.sub(x2).sub(y2).add(z2),
    ], axis=1).reshape([-1, 3, 3]);
    return rotMat;
}

function subtract_flat_id(rot_mats) {
    const rot_nb = int(rot_mats.shape[0]) / 9;
    const id_flat = tf.tile(
        tf.eye(3, dtype=rot_mats.dtype).reshape([1, 9]),
        [rot_mats.shape[0], rot_nb])
    return tf.sub(rot_mats, id_flat);
}

class MANO {
    constructor(
        center_idx=null,
        flat_hand_mean=true,
        ncomps=6,
        side='right',
        mano_root='./data',
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
        if (root_rot_mode == 'axisang') {
            this.rot = 3
        } else {
            throw new Error("Not implemented");
        }

        if (side == 'right') {
            this.mano_path = `${mano_root}/MANO_RIGHT.json`;
        } else if (side == 'left') {
            this.mano_path = `${mano_root}/MANO_LEFT.json`;
        }

        const dd = this.ready_argument(this.mano_path);
        this.betas = dd.betas;
        this.shapedirs = dd.shapedirs;
        this.posedirs = dd.posedirs;
        this.v_template = dd.v_template;
        this.J_regressor = dd.J_regressor;
        this.weights = dd.weights;
        this.faces = tf.cast(dd.f, 'int32');

        const hands_components = dd.hands_components;
        hands_mean = tf.clone(flat_hand_mean ? tf.zeros(hands_components.shape[1]) : dd.hand_mean);
        if (this.use_pca || (this.joint_rot_mode == 'axisang')) {
            this.hands_mean = this.hands_mean
            this.comps = hands_components
            this.select_comps = hands_components.slice([0], [ncomps]);
        } else {
            throw Error;
            // this.hands_mean_rotmat
        }
        
        this.kintree_table = dd.kintree_table;

    }

    async read_mano_json(mano_path) {
        const data = await fetch(mano_path);
        return data.json();
    }

    // set ?
    async ready_argument(mano_path, posekey4vposed = 'pose') {
        const data = await this.read_mano_json(mano_path);

        console.log(Object.keys(data));

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
        console.log(nposeparms);

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
            const v_x = v_shaped.slice([0, 0], [778, 1]);
            const v_y = v_shaped.slice([0, 1], [778, 1]);
            const v_z = v_shaped.slice([0, 2], [778, 1]);
            const J_tmpx = dd.J_regressor.dot(v_x.flatten());
            const J_tmpy = dd.J_regressor.dot(v_y.flatten());
            const J_tmpz = dd.J_regressor.dot(v_z.flatten());
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
     * poses: tensor (48,)
     * betas: tensor (1,) or (10,)
     * trans: tensor (3)

    */
    forward(poses, betas, trans) {
        hand_pose_coeffs = poses.slice([this.rot], [this.ncomps]);
        if (this.use_pca || (this.joint_rot_mode == 'axisang')) {
            if (this.use_pca) {
                full_hand_pose = hand_pose_coeffs.dot(this.select_comps);
            } else {
                full_hand_pose = hand_pose_coeffs;
            }
            full_pose = tf.concat(
                [
                    poses.slice([0], [this.rot]),
                    full_hand_pose.add(this.hands_mean)
                ],
                axis=0
            );

            if (this.root_rot_mode == 'axisang') {
                // pose_map = 
                ;

            } else {
                throw Error;
            }
        } else {
            throw Error;
        }


    }
}



// // Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// // Prepare the model for training: Specify the loss and the optimizer.
// model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// // Train the model using the data.
// model.fit(xs, ys).then(() => {
//     // Use the model to do inference on a data point the model hasn't seen before:
//     model.predict(tf.tensor2d([5], [1, 1])).print();
// });

let mano = new MANO();
const path = './data/MANO_RIGHT.json';
mano.ready_argument(path);

const poses = tf.rand([48]);
const betas = tf.zeros([10]);
const trans = tf.zeros([3]);
mano.forward(poses, betas, trans);

// Rodrigues_forward_test();
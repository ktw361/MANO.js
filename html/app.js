import * as THREE from 'three';
import * as mano from '../mano.js';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/OrbitControls.js';

/*
Reference: 
1. https://github.com/mrdoob/three.js/blob/master/examples/webgl_buffergeometry_indexed.html
*/

window.onload = main;


let camera, scene, renderer, controls;

let mano_model;
let hand_mesh;

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}


async function add_hand_mesh() {
    const mano_obj = await (await fetch('../data/MANO_RIGHT.json')).json();
    mano_model = new mano.MANO(
        mano_obj,
        undefined,
        false,
        6,
        true,
        'axisang',
        'axisang'
    );
    const poses = tf.zeros([1, 9]);
    const betas = tf.zeros([1]);
    const trans = tf.zeros([1, 3]);
    hand_mesh = mano_model.forward_mesh(poses, betas, trans, false);
    scene.add(hand_mesh);
}

function main() {

    function init() {
        camera = new THREE.PerspectiveCamera(27, window.innerWidth / window.innerHeight, 1, 3500);
        camera.position.z = 64;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);

        const light = new THREE.HemisphereLight();
        scene.add(light);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio); // TODO
        document.body.appendChild(renderer.domElement);

        window.addEventListener('resize', onWindowResize);

        controls = new OrbitControls(camera, renderer.domElement); // Note: don't set min/maxDistance!
    }

    function animate() {
        requestAnimationFrame(animate);
        // controls.update();
        render();
    }

    function render() {
        renderer.render(scene, camera);
    }

    init();
    add_hand_mesh();
    animate();

}

document.getElementById('pca0').addEventListener('input', change_pca0);
function change_pca0() {
    const val = parseFloat(this.value);
    let poses = Array(9).fill(0.0);
    poses[3] = val;
    poses = tf.tensor(poses).expandDims(0);
    const betas = tf.zeros([1]);
    const trans = tf.zeros([1, 3]);
    const st = Date.now() / 1000;
    mano_model.set_mesh(hand_mesh, poses, betas, trans, false);
    console.log(`time: ${(Date.now() / 1000) - st}`);

    let sum = 0.0;
    const pos = hand_mesh.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        sum = sum + pos.getX(i);
    }
    console.log(sum);
}
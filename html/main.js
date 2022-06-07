import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/OrbitControls.js';

/*
Reference:
1. https://github.com/mrdoob/three.js/blob/master/examples/webgl_buffergeometry_indexed.html

*/

window.onload = main;


let camera, scene, renderer, controls;
let hand_faces = [];

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}


async function add_hand_mesh() {
    fetch('../data/hand_verts.json')
    .then( resp => resp.json())
    .then( verts => {
        const geometry = new THREE.BufferGeometry();

        const indices = [];
        const vertices = [];
        const colors = [];

        for (let i = 0; i < verts.length; i++) {
            const v = verts[i];
            vertices.push(v[0], v[1], v[2]);
            const r = i / 28 / 28;
            const g = (i % 28) / 28;
            colors.push(r, g, 0.5);
        }

        geometry.setIndex(hand_faces);
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.computeVertexNormals();
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        // geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

        const scale = 100.0;
        geometry.scale(scale, scale, scale);
        // const material = new THREE.MeshBasicMaterial( { color: 0xe0e0e0 });
        const material = new THREE.MeshPhongMaterial( {
            side: THREE.DoubleSide,
            vertexColors: true
        } );

        const mesh = new THREE.Mesh(geometry, material);

        const geo = new THREE.EdgesGeometry(mesh.geometry); // or WireframeGeometry
        const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
        const wireframe = new THREE.LineSegments(geo, mat);
        mesh.add(wireframe);

        // return mesh;
        scene.add(mesh);
    });
}

function main() {

    init();
    // compute_geometry();
    // scene.add ( get_hand_mesh() );
    // const hand_mesh = compute_geometry();
    // scene.add( hand_mesh );

    add_hand_mesh();
    animate();

    function init() {
        // camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 0.01, 10 );
        // camera.position.z = 1;

        // camera = new THREE.PerspectiveCamera(27, window.innerWidth / window.innerHeight, 1, 3500);
        // camera.position.z = 64;

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

        // Load faces
        fetch('../data/hand_faces.json')
            .then(resp => resp.json())
            .then(faces => {
                for (const f of faces) {
                    hand_faces.push(f[0], f[1], f[2]);
                }
            });
    }

    function animate() {
        requestAnimationFrame(animate);
        // controls.update();
        render();
    }

    function render() {
        // const time = Date.now() * 0.001;
        // mesh.rotation.x = time * 0.25;
        // mesh.rotation.y = time * 0.5;
        renderer.render(scene, camera);
    }

}

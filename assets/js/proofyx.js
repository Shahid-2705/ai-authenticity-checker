/**
 * ProofyX -- Dark mode + Three.js neural mesh background.
 *
 * Forces dark mode classes and initializes an interactive
 * particle network background using Three.js or Vanta.js.
 */

(function () {
    "use strict";

    // Force dark mode immediately
    document.documentElement.classList.add("dark");
    document.body.classList.add("dark");
    document.body.style.backgroundColor = "#09090B";

    // Initialize 3D background after CDN scripts load
    setTimeout(function () {
        if (document.getElementById("proofyx-neural-canvas")) return;
        var reducedMotion = window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;
        if (reducedMotion) return;

        if (typeof THREE !== "undefined") {
            initThreeBg();
        } else if (typeof VANTA !== "undefined") {
            initVantaBg();
        }
    }, 600);

    function initThreeBg() {
        var canvas = document.createElement("canvas");
        canvas.id = "proofyx-neural-canvas";
        Object.assign(canvas.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            zIndex: "0",
            pointerEvents: "none",
            opacity: "0.18",
        });
        document.body.prepend(canvas);

        var renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            alpha: true,
            antialias: false,
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        renderer.setSize(window.innerWidth, window.innerHeight);

        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            1,
            1000
        );
        camera.position.z = 300;

        var COUNT = 90;
        var positions = new Float32Array(COUNT * 3);
        var velocities = [];
        for (var i = 0; i < COUNT; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 500;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 500;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
            velocities.push({
                x: (Math.random() - 0.5) * 0.3,
                y: (Math.random() - 0.5) * 0.3,
                z: (Math.random() - 0.5) * 0.1,
            });
        }

        var pointsGeo = new THREE.BufferGeometry();
        pointsGeo.setAttribute(
            "position",
            new THREE.BufferAttribute(positions, 3)
        );
        var pointsMat = new THREE.PointsMaterial({
            color: 0x6366f1,
            size: 2.5,
            transparent: true,
            opacity: 0.7,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });
        scene.add(new THREE.Points(pointsGeo, pointsMat));

        var lineMat = new THREE.LineBasicMaterial({
            color: 0x3f3f46,
            transparent: true,
            opacity: 0.15,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        var mouse = { x: 0, y: 0 };
        document.addEventListener("mousemove", function (e) {
            mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
            mouse.y = -(e.clientY / window.innerHeight - 0.5) * 2;
        });

        var frameCount = 0;
        var THRESHOLD = 80;

        function animate() {
            requestAnimationFrame(animate);
            frameCount++;
            if (frameCount % 2 !== 0) return;

            var pos = pointsGeo.attributes.position.array;
            for (var i = 0; i < COUNT; i++) {
                pos[i * 3] += velocities[i].x + mouse.x * 0.05;
                pos[i * 3 + 1] += velocities[i].y + mouse.y * 0.05;
                pos[i * 3 + 2] += velocities[i].z;
                if (pos[i * 3] > 250) pos[i * 3] = -250;
                if (pos[i * 3] < -250) pos[i * 3] = 250;
                if (pos[i * 3 + 1] > 250) pos[i * 3 + 1] = -250;
                if (pos[i * 3 + 1] < -250) pos[i * 3 + 1] = 250;
            }
            pointsGeo.attributes.position.needsUpdate = true;

            if (frameCount % 6 === 0) {
                scene.children.forEach(function (c) {
                    if (c.isLineSegments) scene.remove(c);
                });
                var lp = [];
                for (var i = 0; i < COUNT; i++) {
                    for (var j = i + 1; j < COUNT; j++) {
                        var dx = pos[i * 3] - pos[j * 3];
                        var dy = pos[i * 3 + 1] - pos[j * 3 + 1];
                        var dz = pos[i * 3 + 2] - pos[j * 3 + 2];
                        if (Math.sqrt(dx * dx + dy * dy + dz * dz) < THRESHOLD) {
                            lp.push(
                                pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2],
                                pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]
                            );
                        }
                    }
                }
                if (lp.length > 0) {
                    var lg = new THREE.BufferGeometry();
                    lg.setAttribute(
                        "position",
                        new THREE.Float32BufferAttribute(lp, 3)
                    );
                    scene.add(new THREE.LineSegments(lg, lineMat));
                }
            }
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener("resize", function () {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    function initVantaBg() {
        var el = document.createElement("div");
        el.id = "proofyx-neural-canvas";
        Object.assign(el.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            zIndex: "0",
            pointerEvents: "none",
            opacity: "0.18",
        });
        document.body.prepend(el);
        VANTA.NET({
            el: el,
            THREE: typeof THREE !== "undefined" ? THREE : undefined,
            mouseControls: true,
            touchControls: false,
            minHeight: 200,
            minWidth: 200,
            scale: 1.0,
            scaleMobile: 0.5,
            color: 0x6366f1,
            backgroundColor: 0x09090b,
            points: 8,
            maxDistance: 20,
            spacing: 16,
        });
    }
})();

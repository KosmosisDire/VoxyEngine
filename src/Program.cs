
using System.Numerics;
using Engine;
using Silk.NET.Input;
using VoxelEngine;


Environment.CurrentDirectory = @"C:\Main Documents\Projects\Coding\C#\VoxelRaymarcher\src";
var window = new Window();
await window.Load();

// lock cursor
var mouse = window.context.Input.Mice[0];
var keyboard = window.context.Input.Keyboards[0];
mouse.Click += (m, b, v) => mouse.Cursor.CursorMode = CursorMode.Disabled;
var camera = new FreeLookCamera(keyboard, mouse, new Vector3(0, 0, 3), new Vector3(0, 0, 0));

var voxelRenderer = new VoxelRenderer(window, camera);
voxelRenderer.UpdateTerrain(Vector3.Zero);

window.OnRender((dt) =>
{
    voxelRenderer.Draw();
});

window.OnUpdate((dt) =>
{
    camera.AspectRatio = window.Size.X / window.Size.Y;
    camera.Update(dt);
});

// using System.Numerics;
// var tree = new OctreeGenerator();
// var positions = tree.GenerateOctree(32, Vector3.Zero).Where(n => n.PosAndSize.W == 8).Select(n => new Vector3(n.PosAndSize.X, n.PosAndSize.Y, n.PosAndSize.Z)).ToList();

// new PointCloudVisualizer(positions, 3).Run();
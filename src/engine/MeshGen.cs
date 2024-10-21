


using Engine;
using Silk.NET.Maths;

public static class MeshGen
{
    public static Mesh Quad(GLContext ctx)
    {
        Vector3D<float>[] vertices = 
        [
            new Vector3D<float>( 1.0f,  1.0f, 0.0f),
            new Vector3D<float>( 1.0f, -1.0f, 0.0f),
            new Vector3D<float>(-1.0f, -1.0f, 0.0f),
            new Vector3D<float>(-1.0f,  1.0f, 0.0f)
        ];
        uint[] indices = [0, 1, 3, 1, 2, 3];
        Vector3D<float>[] normals =
        [
            new(0, 0, 1),
            new(0, 0, 1),
            new(0, 0, 1),
            new(0, 0, 1)
        ];
        Vector2D<float>[] uvs =
        [
            new(1, 1),
            new(1, 0),
            new(0, 0),
            new(0, 1)
        ];

        return new Mesh(ctx, vertices, indices, normals, uvs);
    }
}
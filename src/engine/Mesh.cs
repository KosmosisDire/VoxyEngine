using System.Numerics;
using Silk.NET.Maths;
using Silk.NET.OpenGL;

namespace Engine;

public class Mesh : IDisposable
{
    private uint vao;
    private uint vbo;
    private uint nbo;
    private uint ubo;
    private uint ibo;

    public List<Vector3D<float>> vertices;
    public List<Vector3D<float>> normals;
    public List<Vector2D<float>> uvs;
    public List<uint> indices;

    public int VertexCount => vertices.Count;

    public Bounds WorldBounds { get; private set; }
    private GLContext ctx;

    public Mesh(GLContext ctx, Vector3D<float>[] vertices, uint[] indices, Vector3D<float>[] normals = null, Vector2D<float>[] uvs = null)
    {
        this.ctx = ctx;
        this.vertices = vertices.ToList();
        this.normals = normals?.ToList() ?? new (vertices.Length);
        this.uvs = uvs?.ToList() ?? new (vertices.Length);
        this.indices = indices.ToList();

        ctx.ExecuteCmd((dt, gl) => 
        {
            this.vao = gl.GenVertexArray();
            this.vbo = gl.GenBuffer();
            this.nbo = gl.GenBuffer();
            this.ubo = gl.GenBuffer();
            this.ibo = gl.GenBuffer();

            if (normals == null)
            {
                RecalculateNormals();
            }
            else
            {
                UploadBuffers();
            }
        });
    }

    public Mesh()
    {
        this.vertices = [];
        this.normals = [];
        this.uvs = [];
        this.indices = [];
        this.vao = 0;
        this.vbo = 0;
        this.nbo = 0;
        this.ubo = 0;
        this.ibo = 0;
    }

    public unsafe void Draw()
    {
        ctx.RenderCmd((dt, gl) =>
        {
            gl.BindVertexArray(vao);
            gl.BindBuffer(GLEnum.ElementArrayBuffer, ibo);
            gl.DrawElements(GLEnum.Triangles, indicesCount, GLEnum.UnsignedInt, null);
        });
    }

    public float[] verticesArray;
    public float[] normalsArray;
    public float[] uvsArray;
    public uint[] indicesArray;
    public uint indicesCount;

    public unsafe void CreateFlattenedBuffers() 
    {
        verticesArray = new float[this.vertices.Count * 3];
        normalsArray = new float[this.normals.Count * 3];
        uvsArray = new float[this.uvs.Count * 2];
        indicesArray = new uint[this.indices.Count];
        indicesCount = (uint)this.indices.Count;

        for (int i = 0; i < this.vertices.Count; i++)
        {
            verticesArray[i * 3] = this.vertices[i].X;
            verticesArray[i * 3 + 1] = this.vertices[i].Y;
            verticesArray[i * 3 + 2] = this.vertices[i].Z;
        }

        for (int i = 0; i < this.normals.Count; i++)
        {
            normalsArray[i * 3] = this.normals[i].X;
            normalsArray[i * 3 + 1] = this.normals[i].Y;
            normalsArray[i * 3 + 2] = this.normals[i].Z;
        }

        for (int i = 0; i < this.uvs.Count; i++)
        {
            uvsArray[i * 2] = this.uvs[i].X;
            uvsArray[i * 2 + 1] = this.uvs[i].Y;
        }

        for (int i = 0; i < this.indices.Count; i++)
        {
            indicesArray[i] = this.indices[i];
        }
    }

    public unsafe void UploadBuffers()
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            if (vao == 0)
            {
                vao = gl.GenVertexArray();
                vbo = gl.GenBuffer();
                nbo = gl.GenBuffer();
                ubo = gl.GenBuffer();
                ibo = gl.GenBuffer();
            }

            gl.BindVertexArray(vao);

            gl.BindBuffer(GLEnum.ArrayBuffer, vbo);
            gl.BufferData(GLEnum.ArrayBuffer, (ReadOnlySpan<float>)verticesArray.AsSpan(), GLEnum.StaticDraw);
            gl.VertexAttribPointer(0, 3, GLEnum.Float, false, 3 * sizeof(float), null);
            gl.EnableVertexAttribArray(0);

            gl.BindBuffer(GLEnum.ArrayBuffer, nbo);
            gl.BufferData(GLEnum.ArrayBuffer, (ReadOnlySpan<float>)normalsArray.AsSpan(), GLEnum.StaticDraw);
            gl.VertexAttribPointer(1, 3, GLEnum.Float, false, 3 * sizeof(float), null);
            gl.EnableVertexAttribArray(1);

            gl.BindBuffer(GLEnum.ArrayBuffer, ubo);
            gl.BufferData(GLEnum.ArrayBuffer, (ReadOnlySpan<float>)uvsArray.AsSpan(), GLEnum.StaticDraw);
            gl.VertexAttribPointer(2, 2, GLEnum.Float, false, 2 * sizeof(float), null);
            gl.EnableVertexAttribArray(2);

            gl.BindBuffer(GLEnum.ArrayBuffer, ibo);
            gl.BufferData(GLEnum.ArrayBuffer, (ReadOnlySpan<uint>)indicesArray.AsSpan(), GLEnum.StaticDraw);

            gl.BindBuffer(GLEnum.ArrayBuffer, 0);
            gl.BindVertexArray(0);

            RecalculateBounds();

            vertices.Clear();
            normals.Clear();
            uvs.Clear();
            indices.Clear();

            Console.WriteLine($"Uploaded buffers: Vertices: {verticesArray.Length}, Indices: {indicesArray.Length}");
        });
    }

    public Mesh RecalculateNormals()
    {
        lock (this)
        {
            Vector3D<float>[] normals = new Vector3D<float>[vertices.Count];
            for (int i = 0; i < indices.Count; i += 3)
            {
                Vector3D<float> aB = vertices[(int)indices[i]];
                Vector3D<float> bB = vertices[(int)indices[i + 1]];
                Vector3D<float> cB = vertices[(int)indices[i + 2]];

                Vector3D<float> a = new Vector3D<float>(aB.X, aB.Y, aB.Z);
                Vector3D<float> b = new Vector3D<float>(bB.X, bB.Y, bB.Z);
                Vector3D<float> c = new Vector3D<float>(cB.X, cB.Y, cB.Z);

                Vector3D<float> normal = Vector3D.Normalize(Vector3D.Cross(b - a, c - a));

                normals[indices[i]] += normal;
                normals[indices[i + 1]] += normal;
                normals[indices[i + 2]] += normal;
            }

            for (int i = 0; i < normals.Length; i++)
            {
                normals[i] = Vector3D.Normalize(normals[i]);
            }

            this.normals = normals.ToList();
            UploadBuffers();
        }
        return this;
    }

    public void RecalculateBounds()
    {
        Vector3D<float> min = new Vector3D<float>(float.MaxValue);
        Vector3D<float> max = new Vector3D<float>(float.MinValue);

        foreach (var vertex in vertices)
        {
            min = Vector3D.Min(min, vertex);
            max = Vector3D.Max(max, vertex);
        }

        WorldBounds = new Bounds((Vector3)min, (Vector3)max);
    }

    public void Dispose()
    {
        ctx.ExecuteCmd((dt, gl) => 
        {
            gl.DeleteVertexArray(vao);
            gl.DeleteBuffer(vbo);
            gl.DeleteBuffer(nbo);
            gl.DeleteBuffer(ubo);
            gl.DeleteBuffer(ibo);
            GC.SuppressFinalize(this);
        });
    }
}
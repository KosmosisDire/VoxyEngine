using System.Numerics;
using Engine;
using Silk.NET.OpenGL;

public class BitmaskGenerator : IDisposable
{
    public const int GRID_SIZE = 8;
    public const int DIRECTIONS_COUNT = 8;
    public const int POSITIONS_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;  // 512
    public const int UINTS_PER_BITMASK = 16;  // 4 uvec4s = 16 uints

    private readonly GLContext ctx;
    private uint bitmaskBuffer;
    private bool isInitialized;

    public BitmaskGenerator(GLContext ctx)
    {
        this.ctx = ctx;
        CreateBuffer();
        PrecomputeAndUploadBitmasks();
        isInitialized = true;
    }

    private unsafe void CreateBuffer()
    {
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.GenBuffers(1, out bitmaskBuffer);
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, bitmaskBuffer);
            // Size is: directions * positions * uints * sizeof(uint)
            int bufferSize = DIRECTIONS_COUNT * POSITIONS_COUNT * UINTS_PER_BITMASK * sizeof(uint);
            gl.BufferData(BufferTargetARB.ShaderStorageBuffer, (nuint)bufferSize, null, BufferUsageARB.StaticDraw);
            gl.ObjectLabel(ObjectIdentifier.Buffer, bitmaskBuffer, 20, "Directional Bitmasks");
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public int CellIndex(Vector3I cell)
    {
        return cell.X + cell.Y * GRID_SIZE + cell.Z * GRID_SIZE * GRID_SIZE;
    }

    readonly Vector3I[] directions = 
    [
        new (1, 1, 1),
        new (-1, 1, 1),
        new (1, -1, 1),
        new (-1, -1, 1),
        new (1, 1, -1),
        new (-1, 1, -1),
        new (1, -1, -1),
        new (-1, -1, -1)
    ];

    private unsafe void PrecomputeAndUploadBitmasks()
    {
        // Combined array for all directions and positions
        uint[] allBitmasks = new uint[DIRECTIONS_COUNT * POSITIONS_COUNT * UINTS_PER_BITMASK];

        for (int x = 0; x < GRID_SIZE; x++)
        {
            for (int y = 0; y < GRID_SIZE; y++)
            {
                for (int z = 0; z < GRID_SIZE; z++)
                {
                    Vector3I startCell = new Vector3I(x, y, z);
                    int startIndex = CellIndex(startCell);

                    // For each direction
                    for (int dirIndex = 0; dirIndex < directions.Length; dirIndex++)
                    {
                        Vector3I dir = directions[dirIndex];

                        // Calculate base index for this bitmask in the array
                        // startIndex * 8 gives us the bitmask index (0-4095)
                        // multiply by 16 because each bitmask uses 16 uints
                        int bitmaskBaseIndex = (startIndex * 8 + dirIndex) * 16;

                        // Find the corner of our box
                        Vector3I corner = new Vector3I(
                            dir.X > 0 ? GRID_SIZE - 1 : 0,
                            dir.Y > 0 ? GRID_SIZE - 1 : 0,
                            dir.Z > 0 ? GRID_SIZE - 1 : 0
                        );

                        // For all cells in the box defined by startCell and corner
                        int minX = Math.Min(startCell.X, corner.X);
                        int maxX = Math.Max(startCell.X, corner.X);
                        int minY = Math.Min(startCell.Y, corner.Y);
                        int maxY = Math.Max(startCell.Y, corner.Y);
                        int minZ = Math.Min(startCell.Z, corner.Z);
                        int maxZ = Math.Max(startCell.Z, corner.Z);

                        for (int tx = minX; tx <= maxX; tx++)
                        {
                            for (int ty = minY; ty <= maxY; ty++)
                            {
                                for (int tz = minZ; tz <= maxZ; tz++)
                                {
                                    Vector3I targetCell = new Vector3I(tx, ty, tz);
                                    int targetCellIndex = CellIndex(targetCell);

                                    // Calculate which uint and which bit within that uint
                                    int uintOffset = targetCellIndex / 32;
                                    int bitIndex = targetCellIndex % 32;

                                    // Set the bit in the appropriate uint of this bitmask
                                    allBitmasks[bitmaskBaseIndex + uintOffset] |= 1u << bitIndex;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upload to GPU
        ctx.ExecuteCmd((dt, gl) =>
        {
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, bitmaskBuffer);
            fixed (void* data = allBitmasks)
            {
                gl.BufferSubData(BufferTargetARB.ShaderStorageBuffer, 0,
                    (nuint)(DIRECTIONS_COUNT * POSITIONS_COUNT * UINTS_PER_BITMASK * sizeof(uint)), data);
            }
            gl.BindBuffer(BufferTargetARB.ShaderStorageBuffer, 0);
        });
    }

    public void BindBuffer(GL gl)
    {
        if (!isInitialized) return;
        gl.BindBufferBase(BufferTargetARB.ShaderStorageBuffer, 5, bitmaskBuffer);
    }

    public unsafe void Dispose()
    {
        if (!isInitialized) return;
        ctx.ExecuteCmd((dt, gl) => gl.DeleteBuffer(bitmaskBuffer));
        isInitialized = false;
        GC.SuppressFinalize(this);
    }
}
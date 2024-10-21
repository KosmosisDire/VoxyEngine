using System.Numerics;

namespace Engine;

public class Frustum
{
    public Vector4[] Planes { get; private set; }

    public Frustum()
    {
        Planes = new Vector4[6];
    }

    public void CalculateFrustumPlanes(Matrix4x4 viewProjectionMatrix)
    {
        // Left plane
        Planes[0] = new Vector4(
            viewProjectionMatrix.M14 + viewProjectionMatrix.M11,
            viewProjectionMatrix.M24 + viewProjectionMatrix.M21,
            viewProjectionMatrix.M34 + viewProjectionMatrix.M31,
            viewProjectionMatrix.M44 + viewProjectionMatrix.M41
        );

        // Right plane
        Planes[1] = new Vector4(
            viewProjectionMatrix.M14 - viewProjectionMatrix.M11,
            viewProjectionMatrix.M24 - viewProjectionMatrix.M21,
            viewProjectionMatrix.M34 - viewProjectionMatrix.M31,
            viewProjectionMatrix.M44 - viewProjectionMatrix.M41
        );

        // Top plane
        Planes[2] = new Vector4(
            viewProjectionMatrix.M14 - viewProjectionMatrix.M12,
            viewProjectionMatrix.M24 - viewProjectionMatrix.M22,
            viewProjectionMatrix.M34 - viewProjectionMatrix.M32,
            viewProjectionMatrix.M44 - viewProjectionMatrix.M42
        );

        // Bottom plane
        Planes[3] = new Vector4(
            viewProjectionMatrix.M14 + viewProjectionMatrix.M12,
            viewProjectionMatrix.M24 + viewProjectionMatrix.M22,
            viewProjectionMatrix.M34 + viewProjectionMatrix.M32,
            viewProjectionMatrix.M44 + viewProjectionMatrix.M42
        );

        // Near plane
        Planes[4] = new Vector4(
            viewProjectionMatrix.M13,
            viewProjectionMatrix.M23,
            viewProjectionMatrix.M33,
            viewProjectionMatrix.M43
        );

        // Far plane
        Planes[5] = new Vector4(
            viewProjectionMatrix.M14 - viewProjectionMatrix.M13,
            viewProjectionMatrix.M24 - viewProjectionMatrix.M23,
            viewProjectionMatrix.M34 - viewProjectionMatrix.M33,
            viewProjectionMatrix.M44 - viewProjectionMatrix.M43
        );

        // Normalize the planes
        for (int i = 0; i < 6; i++)
        {
            float length = MathF.Sqrt(Planes[i].X * Planes[i].X + Planes[i].Y * Planes[i].Y + Planes[i].Z * Planes[i].Z);
            Planes[i] /= length;
        }
    }

    public bool IntersectsAABB(Vector3 min, Vector3 max)
    {
        for (int i = 0; i < 6; i++)
        {
            Vector3 positive = new Vector3(
                Planes[i].X > 0 ? max.X : min.X,
                Planes[i].Y > 0 ? max.Y : min.Y,
                Planes[i].Z > 0 ? max.Z : min.Z
            );

            float distance = Vector3.Dot(new Vector3(Planes[i].X, Planes[i].Y, Planes[i].Z), positive) + Planes[i].W;

            if (distance < 0)
            {
                return false;
            }
        }

        return true;
    }
}
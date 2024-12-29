

using System.Numerics;

public struct Vector2I
{
    public int X, Y;

    public Vector2I(int x, int y)
    {
        X = x;
        Y = y;
    }

    public Vector2I(int s)
    {
        X = s;
        Y = s;
    }

    public Vector2I(uint s)
    {
        X = (int)s;
        Y = (int)s;
    }

    public Vector2I(Vector2 v)
    {
        X = (int)v.X;
        Y = (int)v.Y;
    }

    public static Vector2I operator +(Vector2I a, Vector2I b)
    {
        return new Vector2I(a.X + b.X, a.Y + b.Y);
    }

    public static Vector2I operator -(Vector2I a, Vector2I b)
    {
        return new Vector2I(a.X - b.X, a.Y - b.Y);
    }

    public static Vector2I operator *(Vector2I a, int b)
    {
        return new Vector2I(a.X * b, a.Y * b);
    }

    public static Vector2I operator /(Vector2I a, int b)
    {
        return new Vector2I(a.X / b, a.Y / b);
    }

    public static bool operator ==(Vector2I a, Vector2I b)
    {
        return a.X == b.X && a.Y == b.Y;
    }

    public static bool operator !=(Vector2I a, Vector2I b)
    {
        return a.X != b.X || a.Y != b.Y;
    }

    public override bool Equals(object obj)
    {
        return obj is Vector2I i && i == this;
    }

    public override int GetHashCode()
    {
        return X.GetHashCode() ^ Y.GetHashCode();
    }

    public override string ToString()
    {
        return $"({X}, {Y})";
    }
}
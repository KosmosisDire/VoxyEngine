public static class StaticRandom
{
    [ThreadStatic] private static Random _local;
    [ThreadStatic] private static int _lastSeed = 0;
    public static int Seed { get; set; }

    public static int Next()
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return _local.Next();
    }

    public static int Next(int maxValue)
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return _local.Next(maxValue);
    }

    public static int Next(int minValue, int maxValue)
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return _local.Next(minValue, maxValue);
    }

    public static float NextSingle()
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return _local.NextSingle();
    }

    public static double NextDouble()
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return _local.NextDouble();
    }

    public static float Range(float min, float max)
    {
        if (_local == null || _lastSeed != Seed)
        {
            _local = new Random(Seed);
            _lastSeed = Seed;
        }

        return (float)(_local.NextDouble() * (max - min) + min);
    }

}
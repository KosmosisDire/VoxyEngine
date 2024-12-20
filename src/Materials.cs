


using System.Numerics;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Explicit)]
public struct Material
{
    // gradient colors - rgb (0-1), control pos (0-1)
    [FieldOffset(0)] public Vector4 color1 = new Vector4(0, 0, 0, -1);
    [FieldOffset(16)] public Vector4 color2 = new Vector4(0, 0, 0, -1);
    [FieldOffset(32)] public Vector4 color3 = new Vector4(0, 0, 0, -1);
    [FieldOffset(48)] public Vector4 color4 = new Vector4(0, 0, 0, -1);

    [FieldOffset(64)] public float noiseSize;
    [FieldOffset(68)] public float noiseStrength;
    [FieldOffset(72)] public uint texture1Id;
    [FieldOffset(76)] public uint texture2Id;
    [FieldOffset(80)] public float textureBlend;

    [FieldOffset(84)] public bool blendWithNoise;
    [FieldOffset(88)] public bool isPowder;
    [FieldOffset(92)] public bool isLiquid;
    [FieldOffset(96)] public bool isCollidable;

    [FieldOffset(100)] public float shininess;
    [FieldOffset(104)] public float specularStrength;
    [FieldOffset(108)] public float reflectivity;
    [FieldOffset(112)] public float transparency;
    [FieldOffset(116)] public float refractiveIndex;
    [FieldOffset(120)] public float emission;

    [FieldOffset(124)] float p1;

    public Material() { }
}

public static class Materials
{


    public static readonly string[] materialNames = new string[]
    {
        "Invalid",
        "Dirt",
        "Grass",
        "Stone",
        "Black Stone",
        "Ore",
        "Wood",
        "Log",
        "Leaf",
        "Moss",
        "Water",
        "Snow",
        "Glass",
        "Black",
        "Blue",
        "Green",
        "Cyan",
        "Red",
        "Magenta",
        "Yellow",
        "White",
        "Gray",
        "Light Gray"
    };

    public static readonly Material[] materials = new Material[]
    {
    // Invalid
    new Material {
        color1 = new Vector4(1f, 0f, 1f, 0f),
        color2 = new Vector4(0.8f, 0f, 0.8f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.5f,
        specularStrength = 0.5f,
        reflectivity = 0f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 1f,
        isPowder = false,
        isCollidable = true
    },

    // Dirt
    new Material {
        color1 = new Vector4(0.21f, 0.13f, 0.09f, 0f),
        color2 = new Vector4(0.19f, 0.11f, 0.08f, 0.5f),
        color3 = new Vector4(0.23f, 0.14f, 0.10f, 1f),
        noiseSize = 0.5f,
        noiseStrength = 0.1f,
        shininess = 0.1f,
        specularStrength = 0.05f,
        reflectivity = 0f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = true,
        isCollidable = true
    },

    // Grass
    new Material {
        color1 = new Vector4(0.13f, 0.32f, 0.18f, 0f),
        color2 = new Vector4(0.15f, 0.35f, 0.21f, 0.3f),
        color3 = new Vector4(0.11f, 0.29f, 0.16f, 1f),
        noiseSize = 0.2f,
        noiseStrength = 0.03f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Stone
    new Material {
        color1 = new Vector4(0.65f, 0.65f, 0.67f, 0f),
        color2 = new Vector4(0.60f, 0.60f, 0.62f, 0.5f),
        color3 = new Vector4(0.70f, 0.70f, 0.72f, 1f),
        noiseSize = 0.1f,
        noiseStrength = 0.05f,
        shininess = 0.2f,
        specularStrength = 0.3f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Black Stone
    new Material {
        color1 = new Vector4(0.15f, 0.15f, 0.15f, 0f),
        color2 = new Vector4(0.12f, 0.12f, 0.12f, 0.5f),
        color3 = new Vector4(0.18f, 0.18f, 0.18f, 1f),
        noiseSize = 0.15f,
        noiseStrength = 0.04f,
        shininess = 0.4f,
        specularStrength = 0.3f,
        reflectivity = 0.15f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Ore
    new Material {
        color1 = new Vector4(0.15f, 0.65f, 0.55f, 0f),
        color2 = new Vector4(0.12f, 0.60f, 0.50f, 0.4f),
        color3 = new Vector4(0.18f, 0.70f, 0.60f, 0.8f),
        color4 = new Vector4(0.20f, 0.75f, 0.65f, 1f),
        noiseSize = 0.2f,
        noiseStrength = 0.1f,
        shininess = 0.8f,
        specularStrength = 0.7f,
        reflectivity = 0.3f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0.2f,
        isPowder = false,
        isCollidable = true
    },

    // Wood (processed)
    new Material {
        color1 = new Vector4(0.35f, 0.20f, 0.12f, 0f),
        color2 = new Vector4(0.32f, 0.18f, 0.10f, 0.5f),
        color3 = new Vector4(0.38f, 0.22f, 0.14f, 1f),
        noiseSize = 0.1f,
        noiseStrength = 0.1f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Log (raw wood)
    new Material {
        color1 = new Vector4(0.15f, 0.1f, 0.08f, 0f),
        color2 = new Vector4(0.28f, 0.16f, 0.09f, 0.4f),
        color3 = new Vector4(0.32f, 0.19f, 0.11f, 0.8f),
        color4 = new Vector4(0.26f, 0.15f, 0.08f, 1f),
        noiseSize = 0.15f,
        noiseStrength = 0.12f,
        shininess = 0.2f,
        specularStrength = 0.15f,
        reflectivity = 0.05f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Leaf
    new Material {
        color1 = new Vector4(0.12f, 0.28f, 0.15f, 0f),
        color2 = new Vector4(0.14f, 0.32f, 0.17f, 0.3f),
        color3 = new Vector4(0.10f, 0.25f, 0.13f, 0.7f),
        color4 = new Vector4(0.11f, 0.27f, 0.14f, 1f),
        noiseSize = 0.05f,
        noiseStrength = 0.02f,
        shininess = 0.4f,
        specularStrength = 0.3f,
        reflectivity = 0.1f,
        transparency = 0.1f,
        refractiveIndex = 1.1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Moss
    new Material {
        color1 = new Vector4(0.08f, 0.2f, 0.12f, 0f),
        color2 = new Vector4(0.10f, 0.28f, 0.14f, 0.4f),
        color3 = new Vector4(0.09f, 0.23f, 0.13f, 0.8f),
        color4 = new Vector4(0.10f, 0.26f, 0.15f, 1f),
        noiseSize = 0.2f,
        noiseStrength = 0.04f,
        shininess = 0.3f,
        specularStrength = 0.01f,
        reflectivity = 0f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Water
    new Material {
        color1 = new Vector4(0.3f, 0.4f, 0.7f, 0f),
        color2 = new Vector4(0.3f, 0.4f, 0.8f, 0.5f),
        color3 = new Vector4(0.3f, 0.4f, 1.0f, 1f),
        noiseSize = 0.1f,
        noiseStrength = 0.02f,
        shininess = 0.9f,
        specularStrength = 0.95f,
        reflectivity = 0.3f,
        transparency = 0.5f,
        refractiveIndex = 1.3f,
        emission = 0f,
        isPowder = false,
        isLiquid = true,
        isCollidable = false
    },

    // Snow
    new Material {
        color1 = new Vector4(0.95f, 0.95f, 0.97f, 0f),
        color2 = new Vector4(0.92f, 0.92f, 0.94f, 0.5f),
        color3 = new Vector4(0.98f, 0.98f, 1.0f, 1f),
        noiseSize = 0.1f,
        noiseStrength = 0.02f,
        shininess = 0.9f,
        specularStrength = 0.8f,
        reflectivity = 0.3f,
        transparency = 0.1f,
        refractiveIndex = 1.1f,
        emission = 0f,
        isPowder = true,
        isCollidable = true
    },

    // Glass
    new Material {
        color1 = new Vector4(0.90f, 0.90f, 0.92f, 0f),
        color2 = new Vector4(0.92f, 0.92f, 0.94f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 1.0f,
        specularStrength = 2.0f,
        reflectivity = 0.5f,
        transparency = 0.95f,
        refractiveIndex = 1.5f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Basic colors (more muted versions)
    // Black
    new Material {
        color1 = new Vector4(0.05f, 0.05f, 0.05f, 0f),
        color2 = new Vector4(0.07f, 0.07f, 0.07f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Blue
    new Material {
        color1 = new Vector4(0.0f, 0.0f, 0.7f, 0f),
        color2 = new Vector4(0.0f, 0.0f, 0.8f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Green
    new Material {
        color1 = new Vector4(0.0f, 0.7f, 0.0f, 0f),
        color2 = new Vector4(0.0f, 0.8f, 0.0f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Cyan
    new Material {
        color1 = new Vector4(0.0f, 0.7f, 0.7f, 0f),
        color2 = new Vector4(0.0f, 0.8f, 0.8f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Red
    new Material {
        color1 = new Vector4(0.7f, 0.0f, 0.0f, 0f),
        color2 = new Vector4(0.8f, 0.0f, 0.0f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Magenta
    new Material {
        color1 = new Vector4(0.7f, 0.0f, 0.7f, 0f),
        color2 = new Vector4(0.8f, 0.0f, 0.8f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Yellow
    new Material {
        color1 = new Vector4(0.7f, 0.7f, 0.0f, 0f),
        color2 = new Vector4(0.8f, 0.8f, 0.0f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // White
    new Material {
        color1 = new Vector4(0.9f, 0.9f, 0.9f, 0f),
        color2 = new Vector4(1.0f, 1.0f, 1.0f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Gray
    new Material {
        color1 = new Vector4(0.4f, 0.4f, 0.4f, 0f),
        color2 = new Vector4(0.5f, 0.5f, 0.5f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    },

    // Light Gray
    new Material {
        color1 = new Vector4(0.7f, 0.7f, 0.7f, 0f),
        color2 = new Vector4(0.75f, 0.75f, 0.75f, 1f),
        noiseSize = 0f,
        noiseStrength = 0f,
        shininess = 0.3f,
        specularStrength = 0.2f,
        reflectivity = 0.1f,
        transparency = 0f,
        refractiveIndex = 1f,
        emission = 0f,
        isPowder = false,
        isCollidable = true
    }
    };

    public static string GetMaterialName(uint id)
    {
        if (id < 0 || id >= materialNames.Length)
        {
            return "Unknown";
        }
        return materialNames[id];
    }

    public static string GetMaterialName(int id)
    {
        return GetMaterialName((uint)id);
    }
}
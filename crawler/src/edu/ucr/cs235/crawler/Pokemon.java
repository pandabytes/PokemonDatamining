package edu.ucr.cs235.crawler;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/*
 * Plain Old Data Pokemon class
 *
*/
public class Pokemon
{
  private final static String PokemonDataFormat = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s";
  
  private int generation;
  private int pokedex;
  private String name;
  private List<String> types;
  private HashSet<String> abilities;
  private String maleRatio;
  private String femaleRatio;
  private String height;
  private String weight;
  private String category;
  private HashSet<String> eggGroups;
  private int hp;
  private int attack;
  private int defense;
  private int spAttack;
  private int spDefense;
  private int speed;
  private int movesets;
  
  public Pokemon()
  {
    generation = 0;
    pokedex = 0;
    name = null;
    types = new LinkedList<>();
    abilities = new HashSet<>();
    maleRatio = "0%";
    femaleRatio = "0%";
    height = "0.0 m";
    weight = "0.0 kg";
    category = null;
    eggGroups = new HashSet<>();
    hp = 0;
    attack = 0;
    defense = 0;
    spAttack = 0;
    spDefense = 0;
    speed = 0;
    movesets = 0;
  }

  public Pokemon(int generation, int pokedex, String name, List<String> types,
      HashSet<String> abilities, String maleRatio, String femaleRatio, String height, String weight, String category,
      HashSet<String> eggGroups, int hp, int attack, int defense, int spAttack, int spDefense, int speed, int movesets)
  {
    this.generation = generation;
    this.pokedex = pokedex;
    this.name = name;
    this.types = types;
    this.abilities = abilities;
    this.maleRatio = maleRatio;
    this.femaleRatio = femaleRatio;
    this.height = height;
    this.weight = weight;
    this.category = category;
    this.eggGroups = eggGroups;
    this.hp = hp;
    this.attack = attack;
    this.defense = defense;
    this.spAttack = spAttack;
    this.spDefense = spDefense;
    this.speed = speed;
    this.movesets = movesets;
  }
  
  @Override
  public String toString()
  {
    String result = String.format(PokemonDataFormat, generation, pokedex, name, category, types.toString(), abilities.toString(),
                                  maleRatio, femaleRatio, height, weight, eggGroups.toString(), hp, attack, defense,
                                  spAttack, spDefense, speed, movesets);
    return result;
  }

  public int getPokedex()
  {
    return pokedex;
  }
  public void setPokedex(int pokedex)
  {
    this.pokedex = pokedex;
  }
  public String getName()
  {
    return name;
  }
  public void setName(String name)
  {
    this.name = name;
  }
  public List<String> getTypes()
  {
    return types;
  }
  public void setTypes(List<String> types)
  {
    this.types = types;
  }
  public HashSet<String> getAbilities()
  {
    return abilities;
  }
  public void setAbilities(HashSet<String> abilities)
  {
    this.abilities = abilities;
  }
  public String getMaleRatio()
  {
    return maleRatio;
  }
  public void setMaleRatio(String maleRatio)
  {
    this.maleRatio = maleRatio;
  }
  public String getFemaleRatio()
  {
    return femaleRatio;
  }
  public void setFemaleRatio(String femaleRatio)
  {
    this.femaleRatio = femaleRatio;
  }
  public String getHeight()
  {
    return height;
  }
  public void setHeight(String height)
  {
    this.height = height;
  }
  public String getWeight()
  {
    return weight;
  }
  public void setWeight(String weight)
  {
    this.weight = weight;
  }
  public String getCategory()
  {
    return category;
  }
  public void setCategory(String category)
  {
    this.category = category;
  }
  public HashSet<String> getEggGroups()
  {
    return eggGroups;
  }
  public void setEggGroups(HashSet<String> eggGroups)
  {
    this.eggGroups = eggGroups;
  }
  public int getHp()
  {
    return hp;
  }
  public void setHp(int hp)
  {
    this.hp = hp;
  }
  public int getAttack()
  {
    return attack;
  }
  public void setAttack(int attack)
  {
    this.attack = attack;
  }
  public int getDefense()
  {
    return defense;
  }
  public void setDefense(int defense)
  {
    this.defense = defense;
  }
  public int getSpAttack()
  {
    return spAttack;
  }
  public void setSpAttack(int spAttack)
  {
    this.spAttack = spAttack;
  }
  public int getSpDefense()
  {
    return spDefense;
  }
  public void setSpDefense(int spDefense)
  {
    this.spDefense = spDefense;
  }
  public int getSpeed()
  {
    return speed;
  }
  public void setSpeed(int speed)
  {
    this.speed = speed;
  }
  public int getMovesets()
  {
    return movesets;
  }
  public void setMovesets(int movesets)
  {
    this.movesets = movesets;
  }

  public int getGeneration()
  {
    return generation;
  }

  public void setGeneration(int generation)
  {
    this.generation = generation;
  }
}

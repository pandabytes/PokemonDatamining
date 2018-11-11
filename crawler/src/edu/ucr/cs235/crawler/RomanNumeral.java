package edu.ucr.cs235.crawler;

import java.util.HashMap;

public final class RomanNumeral
{
  private final static HashMap<String, Integer> romanNumerals = new HashMap<>();
  static 
  {
    romanNumerals.put("M", 1000);
    romanNumerals.put("CM", 900);
    romanNumerals.put("D", 500);
    romanNumerals.put("CD", 400);
    romanNumerals.put("C", 100);
    romanNumerals.put("XC", 90);
    romanNumerals.put("L", 50);
    romanNumerals.put("XL", 40);
    romanNumerals.put("X", 10);
    romanNumerals.put("IX", 9);
    romanNumerals.put("V", 5);
    romanNumerals.put("IV", 4);
    romanNumerals.put("I", 1);
  }
  
  public static int convertToNumber(String romanNumber)
  {
    int result = 0;
    int prevValue = Integer.MAX_VALUE;
    for (int i = 0; i < romanNumber.length(); i++)
    {
      int value = romanNumerals.get(romanNumber.substring(i, i + 1));
      
      if (value > prevValue)
      {
        result += value - 2*prevValue;
      }
      else
      {
        result += value;
      }
      
      prevValue = value;
    }
    
    return result;
  }
}

#!/usr/bin/env python3
"""
Main entry point for running all homework questions.
Usage: python main.py
"""

import subprocess
import sys

def print_menu():
    """Print the main menu options."""
    print("\n" + "="*50)
    print("16-825 Assignment 1: Rendering Basics with PyTorch3D")
    print("="*50)
    print("Available Questions:")
    print("1. Practicing with Cameras (360-degree renders & Dolly Zoom)")
    print("2. Practicing with Meshes (Tetrahedron & Cube)")
    print("3. Re-texturing a mesh")
    print("4. Camera Transformations")
    print("5. Rendering Generic 3D Representations")
    print("q. Quit")
    print("="*50)

def print_question_1_menu():
    """Print Question 1 submenu."""
    print("\nQuestion 1 - Practicing with Cameras:")
    print("1. 360-degree render of a cow")
    print("2. Dolly zoom effect")
    print("b. Back to main menu")

def print_question_2_menu():
    """Print Question 2 submenu."""
    print("\nQuestion 2 - Practicing with Meshes:")
    print("1. Tetrahedron")
    print("2. Cube")
    print("3. Both shapes")
    print("b. Back to main menu")

def print_question_5_menu():
    """Print Question 5 submenu."""
    print("\nQuestion 5 - Rendering Generic 3D Representations:")
    print("1. Point clouds from RGB-D images")
    print("2. Parametric functions")
    print("3. Implicit surfaces")
    print("b. Back to main menu")

def handle_question_1():
    """Handle Question 1 subparts."""
    while True:
        print_question_1_menu()
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            print("Running 360-degree cow render...")
            subprocess.run([sys.executable, "-m", "starter.q_1_1"])
        elif choice == '2':
            print("Running dolly zoom effect...")
            subprocess.run([sys.executable, "-m", "starter.q_1_2"])
        elif choice == 'b':
            break
        else:
            print("Invalid choice. Please try again.")

def handle_question_2():
    """Handle Question 2 subparts."""
    while True:
        print_question_2_menu()
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            print("Rendering tetrahedron...")
            subprocess.run([sys.executable, "-m", "starter.q_2", "--shape", "tetrahedron"])
        elif choice == '2':
            print("Rendering cube...")
            subprocess.run([sys.executable, "-m", "starter.q_2", "--shape", "cube"])
        elif choice == '3':
            print("Rendering both tetrahedron and cube...")
            subprocess.run([sys.executable, "-m", "starter.q_2", "--shape", "tetrahedron"])
            subprocess.run([sys.executable, "-m", "starter.q_2", "--shape", "cube"])
        elif choice == 'b':
            break
        else:
            print("Invalid choice. Please try again.")

def handle_question_5():
    """Handle Question 5 subparts."""
    while True:
        print_question_5_menu()
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            print("\nPoint clouds from RGB-D images:")
            print("1. First image")
            print("2. Second image") 
            print("3. Union of both images")
            print("4. All three")
            sub_choice = input("Enter your choice: ").strip()
            
            if sub_choice == '1':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "first"])
            elif sub_choice == '2':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "second"])
            elif sub_choice == '3':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "union"])
            elif sub_choice == '4':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "first"])
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "second"])
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "point_cloud", "--which_plant", "union"])
            else:
                print("Invalid choice.")
                
        elif choice == '2':
            print("\nParametric functions:")
            print("1. Torus")
            print("2. Octahedron")
            print("3. Both")
            sub_choice = input("Enter your choice: ").strip()
            
            if sub_choice == '1':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "parametric", "--which_parameter", "torus"])
            elif sub_choice == '2':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "parametric", "--which_parameter", "octahedron"])
            elif sub_choice == '3':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "parametric", "--which_parameter", "torus"])
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "parametric", "--which_parameter", "octahedron"])
            else:
                print("Invalid choice.")
                
        elif choice == '3':
            print("\nImplicit surfaces:")
            print("1. Torus")
            print("2. Octahedron")
            print("3. Both")
            sub_choice = input("Enter your choice: ").strip()
            
            if sub_choice == '1':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "implicit", "--which_implicit", "torus"])
            elif sub_choice == '2':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "implicit", "--which_implicit", "octahedron"])
            elif sub_choice == '3':
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "implicit", "--which_implicit", "torus"])
                subprocess.run([sys.executable, "-m", "starter.q_5", "--render", "implicit", "--which_implicit", "octahedron"])
            else:
                print("Invalid choice.")
                
        elif choice == 'b':
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main program loop."""
    
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            handle_question_1()
        elif choice == '2':
            handle_question_2()
        elif choice == '3':
            print("Running mesh re-texturing...")
            subprocess.run([sys.executable, "-m", "starter.q_3"])
        elif choice == '4':
            print("Running camera transformations...")
            subprocess.run([sys.executable, "-m", "starter.q_4"])
        elif choice == '5':
            handle_question_5()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5 or 'q' to quit.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
def test_install():
    try:
        from rohan import global_imports
        print(f">>> TEST SUCCESSFUL")
    except:
        print(f">>> TEST NOT SUCCESSFUL. Most probabaly missing dependencies.")

test_install()
